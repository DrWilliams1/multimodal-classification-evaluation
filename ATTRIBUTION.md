# Sources and attribution

This portfolio's experiment, evaluation helpers, notebook structure, tests, and verification workflow were written for the project. Conceptual starting point:

- Jay Alammar and Maarten Grootendorst (2024), *Hands-On Large Language Models*, O'Reilly, Chapter 9: Multimodal Large Language Models. Official source: https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter09 . Upstream code is Apache-2.0; this package does not redistribute that chapter notebook.
- Alec Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*. https://arxiv.org/abs/2103.00020 . Official implementation: https://github.com/openai/CLIP (MIT). The project uses the pretrained checkpoint through Transformers, without bundling its weights or copying the CLIP implementation.
- Checkpoint/model card: https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 . The model card describes research uses and limitations; this portfolio is an evaluation, not evidence of operational suitability.
- Alex Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*. CIFAR-10 was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Dataset description and citation: https://www.cs.toronto.edu/~kriz/cifar.html . Pinned data mirror: https://huggingface.co/datasets/uoft-cs/cifar10/tree/0b2714987fa478483af9968de7c934580d0bb9a2 . Labels and image material retain their source terms; no blanket image-rights or commercial-use conclusion is made here.
- API documentation: https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip . The tested implementation pins Transformers 4.57.6 and records the actual environment.

Images and model weights are downloaded separately from these sources. The code license applies to this project's original code; it does not relicense the dataset, annotations, model weights, or cited literature. The five ensemble templates are a declared small experiment-specific set, not the original authors' full prompt ensemble.
