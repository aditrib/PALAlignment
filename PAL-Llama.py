import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# Model for shared mapping function f
class SharedMapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SharedMapping, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.fc(x)

# Transformer-based embedding
class TextEmbedderLLama:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B", device="cpu", token=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
        self.device = device

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

class PreferenceDataset(Dataset):
    def __init__(self, dataframe, text_embedder, user_id_mapping):
        self.dataframe = dataframe
        self.text_embedder = text_embedder
        self.user_id_mapping = user_id_mapping

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        context_text = row['context'][0]['content']
        chosen_text = row['chosen']['content']
        rejected_text = row['rejected']['content']

        context_embed = self.text_embedder.encode([context_text]).to(torch.float32)
        chosen_embed = self.text_embedder.encode([chosen_text]).to(torch.float32)
        rejected_embed = self.text_embedder.encode([rejected_text]).to(torch.float32)

        assert not torch.isnan(context_embed).any(), "NaN in context embedding"
        assert not torch.isnan(chosen_embed).any(), "NaN in chosen embedding"
        assert not torch.isnan(rejected_embed).any(), "NaN in rejected embedding"

        user_id = self.user_id_mapping[row['user_id']]
        label = torch.tensor(1.0 if row['chosen_score'] > row['rejected_score'] else 0.0).to(torch.float32)

        return context_embed.squeeze(0), chosen_embed.squeeze(0), rejected_embed.squeeze(0), user_id, label

class PAL_A(nn.Module):
    def __init__(self, input_dim, output_dim, num_prototypes, num_users):
        super(PAL_A, self).__init__()
        self.shared_mapping = SharedMapping(input_dim, output_dim)
        # Initialize parameters in FP32
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, output_dim) * 0.01)
        self.user_weights = nn.Parameter(torch.rand(num_users, num_prototypes) * 0.01)

    def forward(self, context, choice_embed, user_id):
        f_context = self.shared_mapping(context)
        f_choice = self.shared_mapping(choice_embed)
        if user_id is None:
            a_i = torch.matmul(self.user_weights.mean(dim=0), self.prototypes)
        else:
            a_i = torch.matmul(self.user_weights[user_id], self.prototypes)
        reward = torch.sum(f_choice * a_i, dim=1)
        return reward

def preference_loss(reward_chosen, reward_rejected, label):
    diff = reward_chosen - reward_rejected
    assert not torch.isnan(diff).any(), "NaN in difference of rewards"
    assert not torch.isnan(label).any(), "NaN in labels"
    return torch.nn.functional.binary_cross_entropy_with_logits(diff, label)

def train_model(model, dataloader, optimizer, device, num_epochs=1):
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        for context, chosen, rejected, user_id, label in dataloader:
            # Move data to device without changing dtype
            context = context.to(device)
            chosen = chosen.to(device)
            rejected = rejected.to(device)
            label = label.to(device)
            user_id = user_id.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                reward_chosen = model(context, chosen, user_id)
                reward_rejected = model(context, rejected, user_id)
                loss = preference_loss(reward_chosen, reward_rejected, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')

def training_loop(num_prototypes, num_epochs, batch_size, learning_rate, model_path, grid_search = False, training = True):
    token = "huggingface_tokeyour_n_here"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_embedder = TextEmbedderLLama(
        model_name="meta-llama/Llama-3.1-8B",
        device=device,
        token=token
    )
    input_dim = 4096
    output_dim = 128

    dataset_path = "hf://datasets/MichaelR207/prism_personalized_1023/"
    splits = {
        'train': 'data/train-00000-of-00001.parquet',
    }
    df = pd.read_parquet(dataset_path + splits["train"])

    unique_user_ids = df['user_id'].unique()
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    num_users = len(unique_user_ids)

    if training:
        model = PAL_A(
            input_dim=input_dim,
            output_dim=output_dim,
            num_prototypes=num_prototypes,
            num_users=num_users
        ).to(device)

        print("Training...")
        train_dataset = PreferenceDataset(df[df['split'] == 'train'], text_embedder, user_id_mapping)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        train_model(model, train_dataloader, optimizer, device, num_epochs)

        output_model = "models/" + model_path
        torch.save(model.state_dict(), output_model)

        print(f"Model saved to {output_model}")

    if grid_search:
        results = validate(model_path, text_embedder, user_id_mapping, num_prototypes=5, device=device)
    else:
        results = predict(model_path, text_embedder, user_id_mapping, num_prototypes=5, device=device)
    print(results)




def make_prediction(model, text_embedder, user_id_mapping, context_text, chosen_text, rejected_text, user_id, device):
    model.eval()

    # Encode text embeddings and cast to float32 to match model precision
    context_embed = text_embedder.encode([context_text]).to(device).to(torch.float32)
    chosen_embed = text_embedder.encode([chosen_text]).to(device).to(torch.float32)
    rejected_embed = text_embedder.encode([rejected_text]).to(device).to(torch.float32)

    # Map user_id to index or handle unseen users
    if user_id in user_id_mapping:
        user_index = user_id_mapping[user_id]
        user_index = torch.tensor([user_index], dtype=torch.long).to(device)
    else:
        user_index = None  # Unseen user case

    # Compute rewards
    with torch.no_grad():
        reward_chosen = model(context_embed, chosen_embed, user_index)
        reward_rejected = model(context_embed, rejected_embed, user_index)
        diff = reward_chosen - reward_rejected

    # Predict based on reward difference
    prediction = "chosen" if diff.mean() > 0 else "rejected"
    return prediction == "chosen"


def predict(model_path, text_embedder, user_id_mapping, num_prototypes, device):
    # Load test dataset
    df_test = pd.read_parquet("hf://datasets/MichaelR207/prism_personalized_1023/data/test-00000-of-00001.parquet")
    df_test = df_test[:1000]

    # Load the saved model
    model = PAL_A(
        input_dim=4096,  # Adjust this if your embedding dimension is different
        output_dim=128,
        num_prototypes=num_prototypes,
        num_users=len(user_id_mapping)
    ).to(device)
    model.load_state_dict(torch.load("models/" + model_path))
    model.eval()

    # Calculate accuracy
    correct_predictions = 0
    for idx, row in df_test.iterrows():
        context_text = row['context'][0]['content']
        chosen_text = row['chosen']['content']
        rejected_text = row['rejected']['content']
        user_id = row['user_id']

        is_correct = make_prediction(
            model,
            text_embedder,
            user_id_mapping,
            context_text,
            chosen_text,
            rejected_text,
            user_id,
            device
        )
        correct_predictions += int(is_correct)

    accuracy = correct_predictions / len(df_test)
    results = f"Model used: {model_path}\nAccuracy on test set: {accuracy * 100:.2f}%"
    print(results)
    resPath = f"results/{model_path}.txt"
    with open(resPath, "w") as file:
        file.write(results)
    return results

def validate(model_path, text_embedder, user_id_mapping, num_prototypes, device):
    # Load test dataset
    df_test = pd.read_parquet("hf://datasets/MichaelR207/prism_personalized_1023/data/validation-00000-of-00001.parquet")
    df_test = df_test[:1000]

    # Load the saved model
    model = PAL_A(
        input_dim=4096,  # Adjust this if your embedding dimension is different
        output_dim=128,
        num_prototypes=num_prototypes,
        num_users=len(user_id_mapping)
    ).to(device)
    model.load_state_dict(torch.load("models/" + model_path))
    model.eval()

    # Calculate accuracy
    correct_predictions = 0
    for idx, row in df_test.iterrows():
        context_text = row['context'][0]['content']
        chosen_text = row['chosen']['content']
        rejected_text = row['rejected']['content']
        user_id = row['user_id']

        is_correct = make_prediction(
            model,
            text_embedder,
            user_id_mapping,
            context_text,
            chosen_text,
            rejected_text,
            user_id,
            device
        )
        correct_predictions += int(is_correct)

    accuracy = correct_predictions / len(df_test)
    results = f"Model used: {model_path}\nAccuracy on test set: {accuracy * 100:.2f}%"
    print(results)
    resPath = f"results/{model_path}.txt"
    with open(resPath, "w") as file:
        file.write(results)
    return results

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train PAL_A model with personalized preferences.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer.")
    parser.add_argument("--num_prototypes", type=int, default=5, help="Number of prototypes in the model.")
    parser.add_argument("--output_model", type=str, default="pal_a_model.pt", help="Path to save the trained model.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search")

    args = parser.parse_args()
    output_model = args.output_model
    num_prototypes = args.num_prototypes
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    train = False
    grid_search = False
    if args.train:
        train = True
    if args.grid_search:
        grid_search = True

    training_loop(num_prototypes, num_epochs, batch_size, lr, output_model, grid_search, train)


if __name__ == "__main__":
    main()