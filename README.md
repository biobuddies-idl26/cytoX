# cytoX
**Predicting cytokine-induced transcription factor activity in PBMCs with a from-scratch PyTorch scGen-inspired VAE.**

A PyTorch reimplementation of scGen that predicts how cytokines reshape gene expression and transcription factor (TF) activity in immune cells. Trained on the Human Cytokine Dictionary (90 cytokines across 12 PBMC types), cytoX uses a VAE with latent-space vector arithmetic to predict a held-out cell type's response to a cytokine. It then infers TF activity from the predicted expression using CollecTRI regulons via decoupler, with an optional decoder that outputs TF activity directly.


*Created as part of the requirements for the Introduction to Deep Learning (11-685) Term Project, Fall 2026, Carnegie Mellon University.*
