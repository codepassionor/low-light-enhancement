from transformers import CLIPTokenizer, CLIPTextModel
import torch

def generate_text_embeddings(text, model_name="openai/clip-vit-base-patch32", device="cuda"):
    """
    Generates CLIP embeddings for the input text.

    Parameters:
    - text: A string representing the input text.
    - model_name: The name of the CLIP model to use (default is "openai/clip-vit-base-patch32").
    - device: The device to use (default is "cuda").

    Returns:
    - text_embeddings: The embeddings of the input text with shape (batch_size, seq_length, hidden_size).
    """
    # Initialize the tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)

    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)

    # Generate embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids=input_ids).last_hidden_state

    return text_embeddings.cuda()

# Example usage
if __name__ == '__main__':
    text = "A photo of a cat"
    embeddings = generate_text_embeddings(text)
    print(embeddings.shape)  # (1, 77, 512)
