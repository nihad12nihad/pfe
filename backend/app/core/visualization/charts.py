import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

class ChartGenerator:
    """Classe principale pour la génération de graphiques standards"""
    
    def __init__(self, output_base_dir: str = "resultats"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
    def _prepare_figure(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Initialise une figure avec des paramètres par défaut"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    def _save_figure(self, fig: plt.Figure, filename: str, subfolder: Optional[str] = None) -> str:
        """Sauvegarde standardisée des figures"""
        output_dir = self.output_base_dir / subfolder if subfolder else self.output_base_dir
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / f"{filename}.png"
        fig.savefig(filepath, bbox_inches='tight', dpi=120)
        plt.close(fig)
        return str(filepath)
    
    def _apply_common_styling(self, ax, title: str, xlabel: str, ylabel: str):
        """Applique un style cohérent à tous les graphiques"""
        ax.set_title(title, pad=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor('#f5f5f5')
        plt.tight_layout()

    def create_barplot(self, data: pd.Series, title: str, xlabel: str, ylabel: str) -> str:
        """Génère un barplot standard"""
        fig, ax = self._prepare_figure()
        sns.barplot(x=data.values, y=data.index, ax=ax, palette='viridis')
        self._apply_common_styling(ax, title, xlabel, ylabel)
        filename = f"barplot_{title.replace(' ', '_').lower()}"
        return self._save_figure(fig, filename)

    def create_histogram(self, data: pd.Series, title: str, xlabel: str, bins: int = 30) -> str:
        """Génère un histogramme avec courbe KDE"""
        fig, ax = self._prepare_figure()
        sns.histplot(data, bins=bins, kde=True, color='skyblue', ax=ax)
        self._apply_common_styling(ax, title, xlabel, "Fréquence")
        filename = f"hist_{xlabel.replace(' ', '_').lower()}"
        return self._save_figure(fig, filename)

    def create_boxplot(self, data: pd.Series, title: str, xlabel: str) -> str:
        """Génère un boxplot standard"""
        fig, ax = self._prepare_figure((8, 6))
        sns.boxplot(x=data, color='lightcoral', ax=ax)
        self._apply_common_styling(ax, title, xlabel, "Valeurs")
        filename = f"boxplot_{xlabel.replace(' ', '_').lower()}"
        return self._save_figure(fig, filename)

    def create_heatmap(self, matrix: pd.DataFrame, title: str, fmt: str = ".2f") -> str:
        """Génère une heatmap pour les matrices de corrélation/confusion"""
        fig, ax = self._prepare_figure((10, 8))
        sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=fmt, square=True, ax=ax)
        self._apply_common_styling(ax, title, "", "")
        filename = f"heatmap_{title.replace(' ', '_').lower()}"
        return self._save_figure(fig, filename)

    def create_scatterplot(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> str:
        """Génère un scatter plot avec droite de régression"""
        fig, ax = self._prepare_figure()
        sns.regplot(x=x_col, y=y_col, data=df, 
                   scatter_kws={"color": "blue"}, 
                   line_kws={"color": "red"},
                   ax=ax)
        self._apply_common_styling(ax, title, x_col, y_col)
        filename = f"scatter_{x_col}_{y_col}"
        return self._save_figure(fig, filename)

    def create_countplot(self, df: pd.DataFrame, column: str, title: str) -> str:
        """Génère un countplot pour variables catégorielles"""
        fig, ax = self._prepare_figure()
        sns.countplot(x=column, data=df, palette='muted', ax=ax)
        self._apply_common_styling(ax, title, column, "Count")
        plt.xticks(rotation=45)
        filename = f"countplot_{column}"
        return self._save_figure(fig, filename)

# Instance globale pour une utilisation facile
chart = ChartGenerator()