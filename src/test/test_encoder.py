import unittest
import torch
from models.encoder import Encoder  

class TestEncoder(unittest.TestCase):

    def setUp(self):
        # Définir le périphérique (device) pour le test, par exemple, "cpu" ou "cuda"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_encoder_initialization(self):
        input_size = 100
        hidden_size = 50

        # Créer une instance de la classe Encoder
        encoder = Encoder(input_size, hidden_size, device=self.device)

        # Assertions pour tester l'initialisation
        self.assertIsNotNone(encoder, "L'encodeur n'est pas initialisé")
        self.assertEqual(hidden_size, encoder.hidden_size, "Taille cachée incorrecte")

    def test_encoder_forward(self):
        input_size = 100
        hidden_size = 50

        # Créer une instance de la classe Encoder
        encoder = Encoder(input_size, hidden_size, device=self.device)

        # Initialiser l'état caché
        hidden_state = encoder.initHidden()

        # Créer une séquence d'entrée de test
        input_sequence = torch.randint(0, input_size, (10,))

        # Passe avant à travers l'encodeur
        output, new_hidden_state = encoder(input_sequence)

        # Assertions pour tester la passe avant
        self.assertEqual(torch.Size([1, 1, 2 * hidden_size]), output.shape, "Forme de la sortie incorrecte")
        self.assertEqual(torch.Size([1, 1, 2 * hidden_size]), new_hidden_state.shape, "Forme de l'état caché incorrecte")

        # Vérifier la compatibilité entre la taille de l'embedding et la taille de la séquence d'entrée
        self.assertEqual(encoder.embedding.weight.shape[1], hidden_size, "Taille de l'embedding incorrecte")
        self.assertEqual(input_sequence.shape[1], hidden_size, "Taille de la séquence d'entrée incorrecte")

if __name__ == '__main__':
    unittest.main()
