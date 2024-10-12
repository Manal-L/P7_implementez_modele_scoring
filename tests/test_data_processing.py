import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        # Configuration d'un DataFrame d'exemple
        self.df = pd.DataFrame({
            'A': [1, 2, 3, None],
            'B': [4, None, 6, 8],
            'TARGET': [1, 0, 1, None]
        })

        # Imputation simple pour les tests
        self.df.fillna(self.df.mean(), inplace=True)

    def test_shape_after_imputation(self):
        # Vérifie que le DataFrame a la bonne forme après imputation
        self.assertEqual(self.df.shape, (4, 3))

    def test_no_missing_values(self):
        # Vérifie qu'il n'y a plus de valeurs manquantes
        self.assertFalse(self.df.isnull().values.any())

    def test_standard_scaler(self):
        # Standardisation
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        scaled_data = scaler.fit_transform(self.df[num_cols])

        # Vérifie que la moyenne des colonnes standardisées est proche de 0
        self.assertAlmostEqual(scaled_data.mean(), 0, delta=1e-7)
        # Vérifie que l'écart type des colonnes standardisées est proche de 1
        self.assertAlmostEqual(scaled_data.std(), 1, delta=1e-7)

if __name__ == '__main__':
    unittest.main()
