# Text-to-Image Generation with CLIP and VQGAN


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FaHoB3cubqadSJenWHADSnb3uRRYd5RD#scrollTo=94fdILI2WBUY)

<img width="633" alt="צילום מסך 2023-07-17 ב-16 30 19" src="https://github.com/shaked32003/Text-to-Image-Generation-with-CLIP-and-VQGAN/assets/96596252/81918838-dd9f-4eb4-9277-b8ecdbc2ebc7">

'A dog on the beach Unreal Engine'


## Overview
This project demonstrates how to generate visually coherent images from textual prompts using the CLIP and VQGAN models. By combining the power of natural language processing and computer vision, the models can create stunning images conditioned on the provided text.

## Features
- Generate high-quality images based on textual prompts
- Fine-tune the image generation process through optimization
- Interactive training loop with periodic image display
- Experiment with different prompts, settings, and model architectures

## Parameters
Here are the parameters used in the project:

- `learning_rate`: Learning rate for the optimization process.
- `batch_size`: Batch size for training.
- `wd`: Weight decay for the AdamW optimizer.
- `noise_factor`: Factor for introducing noise to the generated images.
- `w1` and `w2`: Weights for combining the prompt and extra text encodings.
- `total_iter`: Total number of iterations for training.
- `show_step`: Number of iterations between displaying generated images during training.
- `im_shape`: Shape of the output image (width, height, channels).

## Model and Architecture
The project utilizes two main models: CLIP and VQGAN.

### CLIP
CLIP (Contrastive Language-Image Pretraining) is a powerful model that understands both images and text in a joint embedding space. It can encode images and text into a shared latent space, enabling cross-modal retrieval and generation tasks. In this project, CLIP is used to encode the textual prompts by tokenizing the text and extracting its corresponding embedding.

### VQGAN
VQGAN (Vector Quantized Generative Adversarial Network) is a generative model capable of transforming latent vectors into high-quality images. It consists of an encoder, a vector quantizer, and a decoder. The encoder encodes the input image into a latent space, the vector quantizer discretizes the latent space into a codebook, and the decoder reconstructs the image from the quantized latent vectors. In this project, VQGAN is employed to generate images based on the optimized latent vectors.

## Training Process
The training process follows these steps:

1. Load the CLIP and VQGAN models.
2. Initialize the model parameters and optimizer.
3. Encode the textual prompts using CLIP to obtain their corresponding embeddings.
4. Iterate through a training loop for a specified number of iterations.
5. Optimize the model parameters based on the loss function, which involves comparing the generated images with the encoded prompts using cosine similarity. Cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them. By maximizing the cosine similarity between the generated images and the prompts, the model learns to generate images that align with the given text.
6. Display generated images at specified intervals during training to visualize the progress.
7. Store the generated images and corresponding latent vectors for further analysis or use.
8. Clear the GPU memory to release resources.

## Getting Started
To run this project, you can click the "Open in Colab" button at the top of this README. It will open the project notebook in Google Colab, where you can interactively execute the code and generate your own images based on textual prompts.

## Conclusion
This project demonstrates the capabilities of the CLIP and VQGAN models for text-to-image generation. By leveraging the semantic understanding of CLIP and the generative power of VQGAN, visually coherent images can be created from textual descriptions. The project provides flexibility for experimentation and opens up exciting possibilities for creative applications of text-to-image generation.

[Open in Colab](https://colab.research.google.com/drive/1FaHoB3cubqadSJenWHADSnb3uRRYd5RD#scrollTo=94fdILI2WBUY)
