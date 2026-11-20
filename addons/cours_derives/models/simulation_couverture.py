from odoo import fields, models


class SimulationCouverture(models.Model):
    _name = "cours.simulation.couverture"
    _description = "Simulation de couverture (Deep Hedging)"

    name = fields.Char(string="Nom de la simulation", required=True)
    method = fields.Selection(
        [("delta", "Classique Delta"), ("ia", "IA")],
        string="Méthode de couverture", required=True, default="delta")

    # Paramètres de marché
    S0 = fields.Float(string="Prix initial", default=100.0)
    K = fields.Float(string="Strike", default=100.0)
    T = fields.Float(string="Maturité (ans)", default=1.0)
    sigma = fields.Float(string="Volatilité", default=0.2)

    transaction_cost = fields.Float(string="Coût de transaction", default=0.001)
    n_trajectories = fields.Integer(string="Nombre de trajectoires", default=20000)

    # Résultats (remplis plus tard par la simulation)
    pnl_mean = fields.Float(string="Moyenne du P&L")
    pnl_std = fields.Float(string="Écart-type du P&L")
    var_95 = fields.Float(string="VaR 95 % du P&L")