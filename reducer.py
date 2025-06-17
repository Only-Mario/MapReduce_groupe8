#!/usr/bin/env python3
"""
Reducer amélioré pour PageRank MapReduce
Agrège les contributions et calcule les nouveaux scores PageRank
"""
import sys
import logging
from typing import Dict, Set, Optional
from collections import defaultdict

# Configuration du logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class PageRankReducer:
    """
    Classe pour gérer la réduction PageRank avec gestion des nœuds pendants
    """
    
    def __init__(self, damping_factor: float = 0.85):
        self.damping_factor = damping_factor
        self.nodes_data = defaultdict(lambda: {'rank_sum': 0.0, 'outlinks': ''})
        self.all_nodes: Set[str] = set()
        self.total_dangling_rank = 0.0
        
    def process_line(self, line: str) -> None:
        """Traitement parallélisable des lignes"""
        try:
            node, typ, value = line.strip().split('\t', 2)
            self.all_nodes.add(node)
            
            if typ == 'GRAPH':
                self.nodes_data[node]['outlinks'] = value
            elif typ == 'CONTRIB':
                self.nodes_data[node]['rank_sum'] += float(value)
            elif typ == 'DANGLING':
                self.nodes_data[node]['rank_sum'] += float(value)
                self.nodes_data[node]['outlinks'] = ''
        except ValueError:
            logging.error(f"Ligne mal formatée: {line[:50]}...")

            
    def calculate_final_ranks(self) -> Dict[str, float]:
        """Version optimisée du calcul PageRank"""
        num_nodes = len(self.all_nodes)
        if num_nodes == 0:
            return {}

        # Calcul de la somme totale des rangs des nœuds pendants en une passe
        total_dangling_rank = sum(
            data['rank_sum'] for node, data in self.nodes_data.items()
            if not data['outlinks'] and node in self.all_nodes
        )

        # Calcul parallélisable des rangs finaux
        base_rank = (1 - self.damping_factor) / num_nodes
        dangling_contrib = self.damping_factor * (total_dangling_rank / num_nodes)
        
        final_ranks = {
            node: base_rank + self.damping_factor * self.nodes_data[node]['rank_sum'] + dangling_contrib
            for node in self.all_nodes
        }
        
        return final_ranks
    
    def output_results(self, final_ranks: Dict[str, float]) -> None:
        """
        Génère la sortie finale
        """
        for node in sorted(final_ranks.keys()):
            rank = final_ranks[node]
            outlinks = self.nodes_data[node]['outlinks']
            
            if outlinks:
                print(f"{node} {rank:.10f} {outlinks}")
            else:
                print(f"{node} {rank:.10f}")

def main():
    """
    Point d'entrée principal du reducer
    """
    # Récupérer le facteur d'amortissement depuis les arguments
    damping_factor = 0.85
    if len(sys.argv) > 1:
        try:
            damping_factor = float(sys.argv[1])
            if not 0.0 < damping_factor < 1.0:
                logging.error(f"Facteur d'amortissement invalide: {damping_factor}")
                damping_factor = 0.85
        except ValueError:
            logging.error(f"Facteur d'amortissement non numérique: {sys.argv[1]}")
            damping_factor = 0.85
    
    reducer = PageRankReducer(damping_factor)
    
    try:
        # Traiter toutes les lignes d'entrée
        for line in sys.stdin:
            if line.strip():
                reducer.process_line(line)
        
        # Calculer et afficher les résultats finaux
        final_ranks = reducer.calculate_final_ranks()
        reducer.output_results(final_ranks)
        
    except KeyboardInterrupt:
        logging.error("Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erreur fatale dans le reducer: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()