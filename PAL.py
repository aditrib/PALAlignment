import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd

# Model for shared mapping function f
class SharedMapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SharedMapping, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
# Transformer-based embedding
class TextEmbedder:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)  # Mean pooling
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
        
        # Extract text content
        context_text = row['context'][0]['content']
        chosen_text = row['chosen']['content']
        rejected_text = row['rejected']['content']
        
        # Get embeddings for context, chosen, rejected
        context_embed = self.text_embedder.encode([context_text])
        chosen_embed = self.text_embedder.encode([chosen_text])
        rejected_embed = self.text_embedder.encode([rejected_text])
        
        # Map user_id to index
        user_id = self.user_id_mapping[row['user_id']]
        
        # Set preference label
        label = torch.tensor(1.0 if row['chosen_score'] > row['rejected_score'] else 0.0)
        
        return context_embed.squeeze(0), chosen_embed.squeeze(0), rejected_embed.squeeze(0), user_id, label

class PAL_A(nn.Module):
    def __init__(self, input_dim, output_dim, num_prototypes, num_users):
        super(PAL_A, self).__init__()
        self.shared_mapping = SharedMapping(input_dim, output_dim)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, output_dim))
        self.user_weights = nn.Parameter(torch.rand(num_users, num_prototypes))
        self.num_users = num_users

    def forward(self, context, choice_embed, user_id):
        f_context = self.shared_mapping(context)
        f_choice = self.shared_mapping(choice_embed)  # Differentiate chosen/rejected
        if user_id is None:
            a_i = torch.matmul(self.user_weights.mean(dim=0), self.prototypes)
        else:
            a_i = torch.matmul(self.user_weights[user_id], self.prototypes)
        # Calculate similarity for choice and context separately
        reward = torch.sum(f_choice * a_i, dim=1)  # Dot product for similarity
        #print(reward.shape)
        return reward

"""Using the question's context"""
class PAL_A_Contextual(nn.Module):
    def __init__(self, input_dim, output_dim, num_prototypes, num_users):
        super(PAL_A_Contextual, self).__init__()
        self.shared_mapping = SharedMapping(input_dim, output_dim)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, output_dim))
        self.user_weights = nn.Parameter(torch.rand(num_users, num_prototypes))
        self.num_users = num_users

    def forward(self, context, choice_embed, user_id=None):
        f_context = self.shared_mapping(context)
        f_choice = self.shared_mapping(choice_embed)
        
        combined = torch.cat((f_context, f_choice), dim=1)
        context_aware_choice = nn.Linear(2 * f_choice.size(1), f_choice.size(1)).to(context.device)(combined)
        
        if user_id is None:
            a_i = self.prototypes.mean(dim=0)  # For unseen users
        else:
            a_i = torch.matmul(self.user_weights[user_id], self.prototypes)
        
        # Calculate reward based on context-aware choice embedding
        reward = torch.sum(context_aware_choice * a_i, dim=1)
        return reward


def preference_loss(reward_chosen, reward_rejected, label):
    diff = reward_chosen - reward_rejected
    return torch.nn.functional.binary_cross_entropy_with_logits(diff, label)

# Training Loop
def train_model(model, dataloader, optimizer, device, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for context, chosen, rejected, user_id, label in dataloader:
            context, chosen, rejected, label = context.to(device), chosen.to(device), rejected.to(device), label.to(device)
            user_id = user_id.to(device)
            

            reward_chosen = model(context, chosen, user_id)
            reward_rejected = model(context, rejected, user_id)
            loss = preference_loss(reward_chosen, reward_rejected, label)
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

def make_prediction(model, text_embedder, user_id_mapping, context_text, chosen_text, rejected_text, user_id, device):
    model.eval()

    context_embed = text_embedder.encode([context_text]).to(device)
    chosen_embed = text_embedder.encode([chosen_text]).to(device)
    rejected_embed = text_embedder.encode([rejected_text]).to(device)

    # Map user_id to index or handle unseen users
    if user_id in user_id_mapping:
        user_index = user_id_mapping[user_id]
        user_index = torch.tensor([user_index], dtype=torch.long).to(device)
    else:
        user_index = None  # Unseen user case

    # Compute rewards
    reward_chosen = model(context_embed, chosen_embed, user_index)
    reward_rejected = model(context_embed, rejected_embed, user_index)
    diff = reward_chosen - reward_rejected

    # Predict based on reward difference
    prediction = "chosen" if diff.mean() > 0 else "rejected"
    return prediction == "chosen"


def predict(model_path, text_embedder, user_id_mapping, num_prototypes, device):
    # Load test dataset
    df_test = pd.read_parquet("hf://datasets/MichaelR207/prism_personalized_1023/" + 'data/test-00000-of-00001.parquet')

    # Load the saved model
    model = PAL_A(
        input_dim=768,  # DistilBERT embedding dimensions
        output_dim=128,
        num_prototypes=num_prototypes,
        num_users=len(user_id_mapping)
    ).to(device)
    model.load_state_dict(torch.load("models/"+model_path))
    model.eval()

    # Calculate accuracy
    correct_predictions = 0
    for idx, row in df_test.iterrows():
        context_text = row['context'][0]['content']
        chosen_text = row['chosen']['content']
        rejected_text = row['rejected']['content']
        user_id = row['user_id']

        is_correct = make_prediction(model, text_embedder, user_id_mapping, context_text, chosen_text, rejected_text, user_id, device)
        correct_predictions += int(is_correct)

    accuracy = correct_predictions / len(df_test)
    results = "Model used- " + model_path + f"\nAccuracy on test set: {accuracy * 100:.2f}%"
    print(results)
    resPath = f"results/{model_path}.txt"
    with open(resPath, "w") as file:
        file.write(results)
    return results

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train PAL_A model with personalized preferences.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--dataset", type=str, default="prism", help="Dataset name (default: prism).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer.")
    parser.add_argument("--num_prototypes", type=int, default=5, help="Number of prototypes in the model.")
    parser.add_argument("--use_context", action="store_true", help="Flag to use context-aware embeddings.")
    parser.add_argument("--output_model", type=str, default="pal_a_model.pt", help="Path to save the trained model.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--predict", action="store_true", help="Run predictions.")

    args = parser.parse_args()

    splits = {
        'train': 'data/train-00000-of-00001.parquet',
        'validation': 'data/validation-00000-of-00001.parquet',
        'test': 'data/test-00000-of-00001.parquet'
    }
    dataset_path = "hf://datasets/MichaelR207/prism_personalized_1023/"
    df = pd.read_parquet(dataset_path + splits["train"])

    unique_user_ids = df['user_id'].unique()
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    num_users = len(unique_user_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_embedder = TextEmbedder()

    input_dim = 768  # For DistilBERT embedding dimensions
    output_dim = 128

    if args.train:
        print("Training")
        if args.use_context:
            model = PAL_A_Contextual(
                input_dim=input_dim,
                output_dim=output_dim,
                num_prototypes=args.num_prototypes,
                num_users=num_users
            ).to(device)
        else:
            model = PAL_A(
                input_dim=input_dim,
                output_dim=output_dim,
                num_prototypes=args.num_prototypes,
                num_users=num_users
            ).to(device)

        train_dataset = PreferenceDataset(df[df['split'] == 'train'], text_embedder, user_id_mapping)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        train_model(model, train_dataloader, optimizer, device, args.num_epochs)

        # Save the model
        output_model = "models/"+args.output_model
        torch.save(model.state_dict(), output_model)
        print(f"Model saved to {output_model}")
    if args.predict:
        print("Predicting")
        num_prototypes = args.num_prototypes
        predict(args.output_model, text_embedder, user_id_mapping, num_prototypes, device)


if __name__ == "__main__":
    main()