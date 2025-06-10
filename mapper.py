#!/usr/bin/env python3
"""
Mapper amélioré pour PageRank MapReduce
Traite les données d'entrée et calcule les contributions PageRank
"""
import sys
import logging
from typing import List, Tuple, Optional

# Configuration du logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

def parse_line(line: str) -> Tuple[str, float, List[str]]:
    """
    Parse une ligne d'entrée et extrait le nœud, son rang et ses liens
    
    Formats supportés:
    - "node rank link1,link2,link3" (itérations suivantes)
    - "node link1 link2 link3" (première itération)
    - "node" (nœud sans liens)
    """
    tokens = line.strip().split()
    
    if not tokens:
        raise ValueError("Ligne vide")
    
    node = tokens[0]
    
    if len(tokens) == 1:
        # Nœud isolé
        return node, 1.0, []
    
    elif len(tokens) >= 2:
        try:
            # Essayer de parser le deuxième token comme rang
            rank = float(tokens[1])
            
            # Les liens sont dans le troisième token (séparés par des virgules)
            if len(tokens) > 2:
                links = tokens[2].split(',') if tokens[2] else []
                # Nettoyer les liens vides
                links = [link.strip() for link in links if link.strip()]
            else:
                links = []
                
            return node, rank, links
            
        except ValueError:
            # Le deuxième token n'est pas un nombre, donc c'est la première itération
            # Tous les tokens après le premier sont des liens
            rank = 1.0
            links = [link.strip() for link in tokens[1:] if link.strip()]
            return node, rank, links
    
    return node, 1.0, []

def emit_contributions(node: str, rank: float, outlinks: List[str]) -> None:
    """
    Émet les contributions PageRank pour les liens sortants
    """
    # Émettre la structure du graphe (préservation des liens)
    links_str = ','.join(outlinks) if outlinks else ''
    print(f"{node}\t[{links_str}")
    
    # Émettre les contributions PageRank
    if outlinks:
        contribution = rank / len(outlinks)
        for target in outlinks:
            print(f"{target}\t{contribution:.10f}")
    else:
        # Nœud sans liens sortants - contribution distribuée à tous les nœuds
        # (sera géré par le reducer avec la "random jump" probability)
        pass

def process_input() -> None:
    """
    Traite toutes les lignes d'entrée depuis stdin
    """
    line_count = 0
    error_count = 0
    
    for line in sys.stdin:
        line_count += 1
        
        try:
            if not line.strip():
                continue
                
            node, rank, outlinks = parse_line(line)
            emit_contributions(node, rank, outlinks)
            
        except Exception as e:
            error_count += 1
            logging.error(f"Erreur ligne {line_count}: {line.strip()[:50]}... - {str(e)}")
            continue
    
    # Log des statistiques (vers stderr pour ne pas interférer avec la sortie)
    if error_count > 0:
        logging.error(f"Traitement terminé: {line_count} lignes, {error_count} erreurs")

def main():
    """
    Point d'entrée principal du mapper
    """
    try:
        process_input()
    except KeyboardInterrupt:
        logging.error("Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erreur fatale: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()