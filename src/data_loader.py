import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, tickers, period=None, start=None, end=None):
        self.tickers = tickers
        self.period = period
        self.start = start
        self.end = end

    def load_data(self):
        """
        Recupere les donnees historiques des actions pour les tickers donnes.
        A intervalle hebdomadaire.
        Retourne un dictionnaire avec les tickers comme cles et les DataFrames pandas comme
        """
        stock_data = yf.download(self.tickers, period=self.period, start=self.start, end=self.end, interval='1wk')
        # Gère le décalage des colonnes si plusieurs tickers sont fournis
        # 1. Nettoyage des trous (Forward Fill + Drop)
        # 2. Empilement des tickers en lignes (Stack)
        stock_data = stock_data.ffill().dropna().stack()
        
        # On peut renommer l'index pour que ce soit propre
        stock_data.index.names = ['Date', 'Ticker'] 

        return stock_data
    

