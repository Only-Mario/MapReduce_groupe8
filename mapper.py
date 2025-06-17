#!/usr/bin/env python3
"""
Mapper amélioré pour PageRank MapReduce
Traite les données d'entrée et calcule les contributions PageRank
"""
import sys
import logging
from typing import List, Tuple, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='mapper.log',
    filemode='w'
)

def log_error(message):
    logging.error(message)
    print(f"ERROR: {message}", file=sys.stderr)

def parse_line(line: str) -> Tuple[str, float, List[str]]:
    tokens = line.strip().split()
    if not tokens:
        raise ValueError("Ligne vide")
    
    node = tokens[0]
    
    # Cas spécial pour le nœud E sans liens
    if len(tokens) == 1:
        return node, 1.0, []
    
    try:
        # Essayer de lire un format avec rank
        rank = float(tokens[1])
        links = tokens[2].split(',') if len(tokens) > 2 else []
        return node, rank, [link for link in links if link]
    except ValueError:
        # Format initial sans rank
        return node, 1.0, [link for link in tokens[1:] if link]

def emit_contributions(node: str, rank: float, outlinks: List[str]) -> None:
    """Émet les contributions optimisées pour le traitement parallèle"""
    # Structure du graphe (peut être traité en parallèle)
    print(f"{node}\tGRAPH\t{','.join(outlinks) if outlinks else ''}")
    
    # Contributions PageRank (traitement parallèle des cibles)
    if outlinks:
        contribution = rank / len(outlinks)
        for target in outlinks:
            print(f"{target}\tCONTRIB\t{contribution:.15f}")
    else:
        # Marqueur pour nœuds pendants
        print(f"{node}\tDANGLING\t{rank:.15f}")

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