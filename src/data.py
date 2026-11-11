import torch
import math

def simulate_gbm(S0=100.0, r=0.03, sigma=0.2, T=1.0, n_steps=30, n_paths=100000, seed=42):
    """
    Simule des trajectoires de prix selon le Mouvement Brownien Géométrique (GBM).
    
    Paramètres:
    - S0 : Prix initial du sous-jacent.
    - r : Taux sans risque.
    - sigma : Volatilité.
    - T : Maturité (en années, 1.0 = 1 an).
    - n_steps : Nombre de pas de temps (ex: 30 à 252).
    - n_paths : Nombre de trajectoires à générer.
    - seed : Graine aléatoire pour la reproductibilité.
    
    Retourne:
    - S : Tenseur PyTorch de dimension (n_paths, n_steps + 1) contenant les trajectoires.
    """
    # Fixer la graine pour la reproductibilité (très important pour vos tests)
    torch.manual_seed(seed)
    
    dt = T / n_steps

    # Z suit une loi normale centrée réduite N(0,1)
    # Dimension : (nombre de trajectoires, nombre de pas)
    Z = torch.randn(n_paths, n_steps)

    # Calcul des rendements logarithmiques (vectorisé)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z

    # Cumul des rendements log le long de l'axe temporel (dim=1)
    log_prices = math.log(S0) + torch.cumsum(log_returns, dim=1)

    # Revenir aux prix normaux avec l'exponentielle
    prices = torch.exp(log_prices)

    # Ajouter le prix initial S0 à l'instant t=0 pour chaque trajectoire
    S0_tensor = torch.full((n_paths, 1), S0)
    S = torch.cat([S0_tensor, prices], dim=1)

    return S

# --- Génération des datasets (Tâche S1) ---
if __name__ == "__main__":
    # 1. Dataset d'entraînement (100 000 trajectoires)
    print("Génération du dataset d'entraînement...")
    train_data = simulate_gbm(n_paths=100_000, n_steps=30, seed=42)
    print(f"Shape du train_data : {train_data.shape}") # Attendu : (100000, 31)

    # 2. Dataset de test (20 000 trajectoires, avec une graine différente !)
    print("Génération du dataset de test...")
    test_data = simulate_gbm(n_paths=20_000, n_steps=30, seed=99)
    print(f"Shape du test_data : {test_data.shape}") # Attendu : (20000, 31)
    
    # sauvegarder ces tenseurs avec torch.save() 
    #torch.save(train_data, 'data_train.pt')