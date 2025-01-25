import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd

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

class TextEmbedder:
    def __init__(self, model_name="distilbert-base-uncased", device="cpu"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(device)
        self.device = device

    def encode(self, texts):
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens)
            # Mean pool the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

class PreferenceDataset(Dataset):
    """
    A single dataset class that branches its logic based on `dataset_type`.
    - For 'prism', we assume original structure:
        * context: list of length 1 => [ { "content": "..."} ]
        * chosen: dict => { "content": "..." }
        * rejected: dict => { "content": "..." }

    - For 'chatbot_arena', we assume:
        * context: list of messages => [ { "content": "...", "role": ...}, ... ]
        * chosen: list of dict(s) => e.g. [ { "content": "...", "role": ... } ]
        * rejected: single dict => { "content": "...", "role": ... }
    """
    def __init__(self, dataframe, text_embedder, user_id_mapping, dataset_type="chatbot_arena"):
        self.dataframe = dataframe
        self.text_embedder = text_embedder
        self.user_id_mapping = user_id_mapping
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        if self.dataset_type == "prism":
            # Original logic for 'prism'
            # context, chosen, rejected are each [ { "content": "..."} ]
            context_text = row['context'][0]['content']
            chosen_text = row['chosen']['content']
            rejected_text = row['rejected']['content']

        elif self.dataset_type == "chatbot_arena":
            # Chatbot Arena logic
            # context is a list of messages
            # chosen is a list of length 1 with dict(s)
            # rejected is a single dict
            # We’ll simply pick the last message in 'context' for the context_text
            if len(row['context']) > 0:
                context_text = row['context'][-1]['content']
            else:
                context_text = ""

            # chosen is a list, typically length 1
            if len(row['chosen']) > 0:
                chosen_text = row['chosen'][0]['content']
            else:
                chosen_text = ""

            # rejected is a single dict
            rejected_text = row['rejected']['content']

        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        # Compute embeddings
        context_embed = self.text_embedder.encode([context_text])
        chosen_embed = self.text_embedder.encode([chosen_text])
        rejected_embed = self.text_embedder.encode([rejected_text])

        # Map user_id to index
        user_id = self.user_id_mapping[row['user_id']]

        # Label: 1 if chosen_score > rejected_score, else 0
        chosen_score = row["chosen_score"]
        rejected_score = row["rejected_score"]
        label_val = 1.0 if chosen_score > rejected_score else 0.0
        label = torch.tensor(label_val, dtype=torch.float)

        return (
            context_embed.squeeze(0),   # context, shape [768]
            chosen_embed.squeeze(0),    # chosen
            rejected_embed.squeeze(0),  # rejected
            user_id,
            label
        )

class PAL_A(nn.Module):
    def __init__(self, input_dim, output_dim, num_prototypes, num_users):
        super(PAL_A, self).__init__()
        self.shared_mapping = SharedMapping(input_dim, output_dim)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, output_dim))
        self.user_weights = nn.Parameter(torch.rand(num_users, num_prototypes))
        self.num_users = num_users

    def forward(self, context, choice_embed, user_id):
        f_context = self.shared_mapping(context)      # mapped context
        f_choice = self.shared_mapping(choice_embed)  # mapped choice
        if user_id is None:
            # If user_id is unknown, just average user_weights
            a_i = torch.matmul(self.user_weights.mean(dim=0), self.prototypes)
        else:
            a_i = torch.matmul(self.user_weights[user_id], self.prototypes)
        # Dot product for similarity
        reward = torch.sum(f_choice * a_i, dim=1)
        return reward


class PAL_A_Contextual(nn.Module):
    """
    Variation that tries to incorporate context embedding into the final calculation.
    """
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
        # simple MLP that reprojects combined context+choice to same dimension
        projector = nn.Linear(2 * f_choice.size(1), f_choice.size(1)).to(context.device)
        context_aware_choice = projector(combined)

        if user_id is None:
            a_i = self.prototypes.mean(dim=0)  # fallback
        else:
            a_i = torch.matmul(self.user_weights[user_id], self.prototypes)

        # Dot product for reward
        reward = torch.sum(context_aware_choice * a_i, dim=1)
        return reward

def preference_loss(reward_chosen, reward_rejected, label):
    diff = reward_chosen - reward_rejected
    return torch.nn.functional.binary_cross_entropy_with_logits(diff, label)

def train_model(model, dataloader, optimizer, device, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for context, chosen, rejected, user_id, label in dataloader:
            context, chosen, rejected, label = (
                context.to(device),
                chosen.to(device),
                rejected.to(device),
                label.to(device)
            )
            user_id = user_id.to(device)

            reward_chosen = model(context, chosen, user_id)
            reward_rejected = model(context, rejected, user_id)
            loss = preference_loss(reward_chosen, reward_rejected, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

def make_prediction(model, text_embedder, user_id_mapping, context_text, chosen_text, rejected_text, user_id, device):
    model.eval()

    context_embed = text_embedder.encode([context_text]).to(device)
    chosen_embed = text_embedder.encode([chosen_text]).to(device)
    rejected_embed = text_embedder.encode([rejected_text]).to(device)

    # Map user_id to index or handle unseen
    if user_id in user_id_mapping:
        user_index = torch.tensor([user_id_mapping[user_id]], dtype=torch.long).to(device)
    else:
        user_index = None

    # Compute rewards
    reward_chosen = model(context_embed, chosen_embed, user_index)
    reward_rejected = model(context_embed, rejected_embed, user_index)
    diff = reward_chosen - reward_rejected

    # Return True if chosen is predicted better
    return (diff.mean() > 0).item()

def predict(model_path, text_embedder, user_id_mapping, num_prototypes, device, dataset_path, dataset_type):
    """
    Evaluate the model on the 'test' split.
    """
    # Load test dataset
    df_test = pd.read_parquet(dataset_path + 'data/test-00000-of-00001.parquet')

    # Rebuild model
    model = PAL_A(
        input_dim=768,
        output_dim=128,
        num_prototypes=num_prototypes,
        num_users=len(user_id_mapping)
    ).to(device)
    model.load_state_dict(torch.load("models/" + model_path))
    model.eval()

    correct_predictions = 0
    for idx, row in df_test.iterrows():
        if dataset_type == "chatbot_arena":
            # Chatbot Arena indexing
            if len(row['context']) > 0:
                context_text = row['context'][-1]['content']
            else:
                context_text = ""
            if len(row['chosen']) > 0:
                chosen_text = row['chosen'][0]['content']
            else:
                chosen_text = ""
            rejected_text = row['rejected']['content']
        else:
            # Prism indexing
            context_text = row['context'][0]['content']
            chosen_text = row['chosen']['content']
            rejected_text = row['rejected']['content']

        user_id = row['user_id']
        is_chosen = make_prediction(
            model, text_embedder, user_id_mapping,
            context_text, chosen_text, rejected_text,
            user_id, device
        )
        # Ground truth
        label = 1.0 if row["chosen_score"] > row["rejected_score"] else 0.0
        if is_chosen == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(df_test)
    results = (
        f"Model used: {model_path}\n"
        f"Accuracy on test set: {accuracy * 100:.2f}%"
    )
    print(results)
    with open(f"results/{model_path}.txt", "w") as f:
        f.write(results)
    return results

def validate(model_path, text_embedder, user_id_mapping, num_prototypes, device, dataset_path, dataset_type):
    """
    Evaluate the model on the 'validation' split.
    """
    df_val = pd.read_parquet(dataset_path + 'data/validation-00000-of-00001.parquet')

    model = PAL_A(
        input_dim=768,
        output_dim=128,
        num_prototypes=num_prototypes,
        num_users=len(user_id_mapping)
    ).to(device)
    model.load_state_dict(torch.load("models/" + model_path))
    model.eval()

    correct_predictions = 0
    for idx, row in df_val.iterrows():
        if dataset_type == "chatbot_arena":
            if len(row['context']) > 0:
                context_text = row['context'][-1]['content']
            else:
                context_text = ""
            if len(row['chosen']) > 0:
                chosen_text = row['chosen'][0]['content']
            else:
                chosen_text = ""
            rejected_text = row['rejected']['content']
        else:
            context_text = row['context'][0]['content']
            chosen_text = row['chosen']['content']
            rejected_text = row['rejected']['content']

        user_id = row['user_id']
        is_chosen = make_prediction(
            model, text_embedder, user_id_mapping,
            context_text, chosen_text, rejected_text,
            user_id, device
        )
        label = 1.0 if row["chosen_score"] > row["rejected_score"] else 0.0
        if is_chosen == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(df_val)
    results = (
        f"Model used: {model_path}\n"
        f"Accuracy on validation set: {accuracy * 100:.2f}%"
    )
    print(results)
    with open(f"results/{model_path}.txt", "w") as f:
        f.write(results)
    return results

def main():
    parser = argparse.ArgumentParser(description="Train PAL_A model with personalized preferences.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--dataset", type=str, default="chatbot_arena",
                        help="Which dataset: 'prism' or 'chatbot_arena'.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer.")
    parser.add_argument("--num_prototypes", type=int, default=5, help="Number of prototypes in the model.")
    parser.add_argument("--use_context", action="store_true", help="Use context-aware embeddings.")
    parser.add_argument("--output_model", type=str, default="pal_a_model.pt", help="Name of saved model.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--validate", action="store_true", help="Validate the model.")
    parser.add_argument("--predict", action="store_true", help="Run predictions on the test set.")
    args = parser.parse_args()

    # Decide on dataset paths
    dataset_paths = {
        "prism": "hf://datasets/MichaelR207/prism_personalized_1023/",
        "chatbot_arena": "hf://datasets/MichaelR207/chatbot_arena_personalized_0120/"
    }

    if args.dataset not in dataset_paths:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_path = dataset_paths[args.dataset]
    dataset_type = args.dataset  # pass to our dataset logic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, dataset: {dataset_type}")

    # Initialize text embedder
    text_embedder = TextEmbedder(device=device)

    # Load training data
    train_df_path = dataset_path + "data/train-00000-of-00001.parquet"
    df = pd.read_parquet(train_df_path)

    # Build user ID mapping
    unique_user_ids = df['user_id'].unique()
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    num_users = len(unique_user_ids)

    # Model dimensions
    input_dim = 768  # DistilBERT embedding dimension
    output_dim = 128

    # Potentially create context-based or standard model
    if args.use_context:
        model_cls = PAL_A_Contextual
    else:
        model_cls = PAL_A

    if args.train:
        print("Training...")
        model = model_cls(
            input_dim=input_dim,
            output_dim=output_dim,
            num_prototypes=args.num_prototypes,
            num_users=num_users
        ).to(device)

        # Filter only the training rows if your dataset has a 'split' column
        # Otherwise, the entire df is training
        if 'split' in df.columns:
            train_df = df[df['split'] == 'train']
        else:
            train_df = df

        # Create the dataset & loader
        train_dataset = PreferenceDataset(
            dataframe=train_df,
            text_embedder=text_embedder,
            user_id_mapping=user_id_mapping,
            dataset_type=dataset_type
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        train_model(model, train_dataloader, optimizer, device, args.num_epochs)

        # Save the trained model
        output_model_path = "models/" + args.output_model
        torch.save(model.state_dict(), output_model_path)
        print(f"Model saved to {output_model_path}")

    # Validation
    if args.validate:
        print("Validating on validation split...")
        validate(
            model_path=args.output_model,
            text_embedder=text_embedder,
            user_id_mapping=user_id_mapping,
            num_prototypes=args.num_prototypes,
            device=device,
            dataset_path=dataset_path,
            dataset_type=dataset_type
        )

    # Prediction (test split)
    if args.predict:
        print("Predicting on test split...")
        predict(
            model_path=args.output_model,
            text_embedder=text_embedder,
            user_id_mapping=user_id_mapping,
            num_prototypes=args.num_prototypes,
            device=device,
            dataset_path=dataset_path,
            dataset_type=dataset_type
        )


if __name__ == "__main__":
    main()
