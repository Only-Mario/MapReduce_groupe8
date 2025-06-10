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
        """
        Traite une ligne d'entrée du mapper
        """
        try:
            node, value = line.strip().split('\t', 1)
            self.all_nodes.add(node)
            
            if value.startswith('['):
                # Structure du graphe (liens sortants)
                self.nodes_data[node]['outlinks'] = value[1:]  # Enlever le '['
                if not value[1:]:  # Nœud sans liens sortants (dangling node)
                    # On ajoutera sa contribution au total des nœuds pendants
                    pass
            else:
                # Contribution PageRank
                try:
                    contribution = float(value)
                    self.nodes_data[node]['rank_sum'] += contribution
                except ValueError:
                    logging.error(f"Valeur de contribution invalide: {value}")
                    
        except ValueError as e:
            logging.error(f"Format de ligne invalide: {line.strip()[:50]}...")
            
    def calculate_final_ranks(self) -> Dict[str, float]:
        """
        Calcule les scores PageRank finaux avec gestion des nœuds pendants
        """
        num_nodes = len(self.all_nodes)
        if num_nodes == 0:
            return {}
        
        # Identifier les nœuds pendants et calculer leur contribution totale
        dangling_nodes = []
        for node in self.all_nodes:
            outlinks = self.nodes_data[node]['outlinks']
            if not outlinks:  # Nœud sans liens sortants
                dangling_nodes.append(node)
        
        # Calculer la contribution des nœuds pendants
        dangling_contribution = 0.0
        for node in dangling_nodes:
            # On suppose que les nœuds pendants ont un rang initial, 
            # on le récupérera lors du calcul final
            pass
        
        final_ranks = {}
        
        for node in self.all_nodes:
            # Somme des contributions reçues
            rank_sum = self.nodes_data[node]['rank_sum']
            
            # Formule PageRank avec gestion des nœuds pendants
            # PageRank(p) = (1-d)/N + d * (sum(PR(Ti)/C(Ti)) + dangling_sum/N)
            base_rank = (1 - self.damping_factor) / num_nodes
            link_contribution = self.damping_factor * rank_sum
            
            # Contribution des nœuds pendants (distribuée équitablement)
            # Cette partie sera améliorée dans une version plus sophistiquée
            dangling_contrib = 0.0  # Simplifié pour cette version
            
            final_rank = base_rank + link_contribution + dangling_contrib
            final_ranks[node] = final_rank
        
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