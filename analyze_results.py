#!/usr/bin/env python3
"""
Analyseur de résultats PageRank
Compare les résultats avec NetworkX et génère des métriques détaillées
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class PageRankAnalyzer:
    """
    Classe pour analyser et comparer les résultats PageRank
    """
    
    def __init__(self, input_file: str, results_dir: str = "pagerank_results"):
        self.input_file = input_file
        self.results_dir = Path(results_dir)
        self.graph = None
        self.nx_pagerank = None
        self.mapreduce_results = {}
        
    def load_graph(self):
        """
        Charge le graphe depuis le fichier d'entrée
        """
        self.graph = nx.DiGraph()
        
        with open(self.input_file, 'r') as f:
            for line in f:
                nodes = line.strip().split()
                if len(nodes) > 1:
                    source = nodes[0]
                    for target in nodes[1:]:
                        self.graph.add_edge(source, target)
                elif len(nodes) == 1:
                    self.graph.add_node(nodes[0])
    
    def compute_networkx_pagerank(self, damping_factor: float = 0.85, max_iter: int = 100):
        """
        Calcule PageRank avec NetworkX pour comparaison
        """
        try:
            self.nx_pagerank = nx.pagerank(self.graph, 
                                         alpha=damping_factor, 
                                         max_iter=max_iter,
                                         tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            print("⚠️  NetworkX PageRank n'a pas convergé, utilisation de plus d'itérations")
            self.nx_pagerank = nx.pagerank(self.graph, 
                                         alpha=damping_factor, 
                                         max_iter=max_iter*2,
                                         tol=1e-4)
    
    def load_mapreduce_results(self):
        """
        Charge les résultats MapReduce depuis les fichiers d'itération
        """
        iteration_files = list(self.results_dir.glob("iteration*.txt"))
        iteration_files.sort(key=lambda x: int(x.stem.replace("iteration", "")))
        
        for file in iteration_files:
            iteration_num = int(file.stem.replace("iteration", ""))
            ranks = {}
            
            with open(file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            node = parts[0]
                            rank = float(parts[1])
                            ranks[node] = rank
                        except ValueError:
                            continue
            
            self.mapreduce_results[iteration_num] = ranks
    
    def compare_results(self) -> Dict:
        """
        Compare les résultats MapReduce avec NetworkX
        """
        if not self.nx_pagerank or not self.mapreduce_results:
            return {}
        
        # Prendre la dernière itération
        final_iteration = max(self.mapreduce_results.keys())
        mr_ranks = self.mapreduce_results[final_iteration]
        
        comparison = {
            'final_iteration': final_iteration,
            'nodes_compared': 0,
            'correlation': 0.0,
            'rmse': 0.0,
            'max_difference': 0.0,
            'mean_difference': 0.0,
            'differences': {},
            'rankings_comparison': []
        }
        
        # Comparer les valeurs pour les nœuds communs
        common_nodes = set(self.nx_pagerank.keys()) & set(mr_ranks.keys())
        
        if not common_nodes:
            return comparison
        
        nx_values = [self.nx_pagerank[node] for node in common_nodes]
        mr_values = [mr_ranks[node] for node in common_nodes]
        
        # Calculer les métriques
        comparison['nodes_compared'] = len(common_nodes)
        comparison['correlation'] = np.corrcoef(nx_values, mr_values)[0, 1]
        comparison['rmse'] = np.sqrt(np.mean([(nx - mr)**2 for nx, mr in zip(nx_values, mr_values)]))
        
        differences = [abs(nx - mr) for nx, mr in zip(nx_values, mr_values)]
        comparison['max_difference'] = max(differences)
        comparison['mean_difference'] = np.mean(differences)
        comparison['differences'] = {node: abs(self.nx_pagerank[node] - mr_ranks[node]) for node in common_nodes}
        comparison['rankings_comparison'] = sorted(common_nodes, key=lambda x: (mr_ranks[x], self.nx_pagerank[x]), reverse=True)
        return comparison
    
    def plot_results(self, comparison: Dict):
        """
        Génère des visualisations des résultats de comparaison
        """
        if not comparison:
            print("Aucune donnée à visualiser.")
            return

        # Visualiser les différences
        plt.figure(figsize=(12, 6))
        plt.bar(comparison['differences'].keys(), comparison['differences'].values())
        plt.title("Différences de PageRank (NetworkX vs MapReduce)")
        plt.xlabel("Nœuds")
        plt.ylabel("Différence de PageRank")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        # Visualiser la corrélation
        plt.figure(figsize=(8, 8))
        plt.scatter(comparison['mean_difference'], comparison['correlation'])
        plt.title("Corrélation entre les différences de PageRank et les valeurs de NetworkX")
        plt.xlabel("Différence de PageRank (Mean)")
        plt.ylabel("Corrélation")
        plt.grid()
        plt.tight_layout()
        plt.show()
        
    def save_comparison(self, comparison: Dict, output_file: str = "pagerank_comparison.json"):
        """
        Enregistre les résultats de comparaison dans un fichier JSON
        """
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=4)
        print(f"Résultats de comparaison enregistrés dans {output_file}")

    def run(self, damping_factor: float = 0.85, max_iter: int = 100):
        """
        Exécute l'analyse de PageRank
        """
        self.load_mapreduce_results()
        self.compare_results()
        self.plot_results(self.comparison)
        self.save_comparison(self.comparison)
        self.compute_networkx_pagerank(damping_factor, max_iter)
        self.load_graph()
        self.run(damping_factor, max_iter)
        self.comparison = self.compare_results()
        self.plot_results(self.comparison)
        self.save_comparison(self.comparison)
        
def main():
    analyzer = PageRankAnalyzer()
    analyzer.run()
    """
    Point d'entrée principal pour l'analyse des résultats PageRank
    """
    parser = argparse.ArgumentParser(description="Analyseur de résultats PageRank")
    parser.add_argument("input_file", type=str, help="Fichier d'entrée contenant le graphe")
    parser.add_argument("--results_dir", type=str, default="pagerank_results", help="Répertoire des résultats MapReduce")
    parser.add_argument("--damping_factor", type=float, default=0.85, help="Facteur d'amortissement pour PageRank")
    parser.add_argument("--max_iter", type=int, default=100, help="Nombre maximum d'itérations pour PageRank")
    args = parser.parse_args()
    analyzer = PageRankAnalyzer(args.input_file, args.results_dir)
    analyzer.run(args.damping_factor, args.max_iter) 
    print("Analyse des résultats PageRank terminée.")
if __name__ == "__main__":
    main()
    
    