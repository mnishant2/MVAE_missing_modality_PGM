import numpy as np
from config import *
from paths import *

if __name__ == "main":
  
    with open(os.path.join(lookup_embd_dir, "lookup_mnist_language_id.npy"), 'wb') as f:
      np.save(f, np.random.rand(num_mnist_languages, mnist_language_embd_dims))

    with open(os.path.join(lookup_embd_dir, "lookup_speaker_id.npy"), 'wb') as f:
      np.save(f, np.random.rand(num_speakers, speaker_embd_dims))

    with open(os.path.join(lookup_embd_dir, "lookup_digit.npy"), 'wb') as f:
      np.save(f, np.random.rand(num_digits, digit_embd_dims))

    with open(os.path.join(lookup_embd_dir, "emb_matrix_synthetic_mu.npy"), 'wb') as f:
      np.save(f, np.random.rand(mnist_language_embd_dims + speaker_embd_dims + digit_embd_dims, 
                                synthetic_modality_dims))

    with open(os.path.join(lookup_embd_dir, "emb_matrix_synthetic_sigma.npy"), 'wb') as f:
      np.save(f, np.random.rand(mnist_language_embd_dims + speaker_embd_dims + digit_embd_dims, 
                                synthetic_modality_dims))