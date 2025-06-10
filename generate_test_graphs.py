#!/usr/bin/env python3
"""
Générateur de graphes de test pour l'analyse PageRank
Simule différents types de structures web
"""

import random
import sys
import argparse
from typing import List, Dict, Set, Tuple
import networkx as nx

class WebGraphGenerator:
    """
    Générateur de graphes simulant des structures web réelles
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialise le générateur avec une graine pour la reproductibilité
        """
        random.seed(seed)
        self.graph = {}
    
    def add_node(self, node: str, links: List[str] = None):
        """
        Ajoute un nœud avec ses liens sortants
        """
        if links is None:
            links = []
        self.graph[node] = links
    
    def generate_random_graph(self, num_nodes: int, edge_probability: float = 0.1) -> Dict[str, List[str]]:
        """
        Génère un graphe aléatoire avec probabilité d'arête donnée
        """
        nodes = [f"Page_{i:03d}" for i in range(num_nodes)]
        graph = {}
        
        for node in nodes:
            graph[node] = []
            for target in nodes:
                if node != target and random.random() < edge_probability:
                    graph[node].append(target)
        
        return graph
    
    def generate_scale_free_graph(self, num_nodes: int, m: int = 2) -> Dict[str, List[str]]:
        """
        Génère un graphe scale-free (loi de puissance) typique du web
        Utilise le modèle Barabási-Albert
        """
        # Utiliser NetworkX pour générer le graphe
        G = nx.barabasi_albert_graph(num_nodes, m, seed=42)
        
        # Convertir en graphe dirigé
        directed_G = nx.DiGraph()
        for edge in G.edges():
            # Ajouter les arêtes dans les deux directions avec probabilité différente
            directed_G.add_edge(f"Page_{edge[0]:03d}", f"Page_{edge[1]:03d}")
            if random.random() < 0.3:  # 30% de chance d'avoir un lien retour
                directed_G.add_edge(f"Page_{edge[1]:03d}", f"Page_{edge[0]:03d}")
        
        # Convertir au format attendu
        graph = {}
        for node in directed_G.nodes():
            graph[node] = list(directed_G.successors(node))
        
        return graph
    
    def generate_hub_and_spoke(self, num_hubs: int = 3, pages_per_hub: int = 10) -> Dict[str, List[str]]:
        """
        Génère un graphe hub-and-spoke (centres et rayons)
        Simule des sites avec pages principales et sous-pages
        """
        graph = {}
        
        # Créer les hubs principaux
        hubs = [f"Hub_{i:02d}" for i in range(num_hubs)]
        
        # Interconnecter les hubs
        for hub in hubs:
            other_hubs = [h for h in hubs if h != hub]
            graph[hub] = random.sample(other_hubs, min(len(other_hubs), 2))
        
        # Créer les pages pour chaque hub
        for i, hub in enumerate(hubs):
            hub_pages = [f"Page_{hub}_{j:03d}" for j in range(pages_per_hub)]
            
            # Chaque page du hub pointe vers le hub et quelques autres pages
            for page in hub_pages:
                links = [hub]
                # Ajouter quelques liens internes
                other_pages = [p for p in hub_pages if p != page]
                if other_pages:
                    links.extend(random.sample(other_pages, min(len(other_pages), 3)))
                
                # Parfois, lien vers un autre hub
                if random.random() < 0.1:
                    other_hub = random.choice([h for h in hubs if h != hub])
                    links.append(other_hub)
                
                graph[page] = links
            
            # Le hub pointe vers toutes ses pages
            graph[hub].extend(hub_pages)
        
        return graph
    
    def generate_hierarchical_graph(self, depth: int = 3, branching_factor: int = 3) -> Dict[str, List[str]]:
        """
        Génère un graphe hiérarchique (arbre + liens croisés)
        Simule la structure d'un site web avec navigation
        """
        graph = {}
        
        def create_level(level: int, parent: str = None, node_id: int = 0):
            if level > depth:
                return node_id
            
            # Créer les nœuds de ce niveau
            level_nodes = []
            for i in range(branching_factor ** (level - 1)):
                node = f"Level_{level}_Node_{node_id:03d}"
                level_nodes.append(node)
                graph[node] = []
                node_id += 1
            
            # Liens vers le parent
            if parent:
                for node in level_nodes:
                    graph[node].append(parent)
            
            # Liens horizontaux (entre nœuds du même niveau)
            for i, node in enumerate(level_nodes):
                # Lien vers le nœud suivant (cyclique)
                next_node = level_nodes[(i + 1) % len(level_nodes)]
                graph[node].append(next_node)
            
            # Récursion pour les niveaux inférieurs
            for node in level_nodes:
                node_id = create_level(level + 1, node, node_id)
            
            return node_id
        
        # Créer la racine
        root = "Root"
        graph[root] = []
        create_level(1, root, 0)
        
        return graph
    
    def generate_social_network(self, num_users: int = 50, avg_connections: int = 5) -> Dict[str, List[str]]:
        """
        Génère un graphe simulant un réseau social
        """
        users = [f"User_{i:03d}" for i in range(num_users)]
        graph = {}
        
        for user in users:
            # Nombre de connexions suivant une distribution normale
            num_connections = max(1, int(random.normalvariate(avg_connections, avg_connections / 3)))
            num_connections = min(num_connections, num_users - 1)
            
            # Sélectionner les connexions
            possible_connections = [u for u in users if u != user]
            connections = random.sample(possible_connections, num_connections)
            
            graph[user] = connections
        
        return graph
    
    def add_noise_and_dead_ends(self, graph: Dict[str, List[str]], 
                                dead_end_ratio: float = 0.1, 
                                noise_ratio: float = 0.05) -> Dict[str, List[str]]:
        """
        Ajoute du bruit et des culs-de-sac au graphe pour le rendre plus réaliste
        """
        nodes = list(graph.keys())
        
        # Ajouter des culs-de-sac (pages sans liens sortants)
        num_dead_ends = int(len(nodes) * dead_end_ratio)
        dead_ends = random.sample(nodes, num_dead_ends)
        
        for node in dead_ends:
            if random.random() < 0.7:  # 70% de chance de devenir un vrai cul-de-sac
                graph[node] = []
        
        # Ajouter du bruit (liens aléatoires)
        for node in nodes:
            if random.random() < noise_ratio:
                # Ajouter un lien aléatoire
                possible_targets = [n for n in nodes if n != node and n not in graph[node]]
                if possible_targets:
                    graph[node].append(random.choice(possible_targets))
        
        return graph
    
    def save_graph(self, graph: Dict[str, List[str]], filename: str):
        """
        Sauvegarde le graphe dans un fichier au format attendu
        """
        with open(filename, 'w') as f:
            for node, links in graph.items():
                if links:
                    f.write(f"{node} {' '.join(links)}\n")
                else:
                    f.write(f"{node}\n")
    
    def generate_test_suite(self, output_dir: str = "."):
        """
        Génère une suite complète de graphes de test
        """
        test_cases = [
            ("small_random", self.generate_random_graph(10, 0.2)),
            ("medium_scale_free", self.generate_scale_free_graph(50, 3)),
            ("hub_spoke", self.generate_hub_and_spoke(4, 8)),
            ("hierarchical", self.generate_hierarchical_graph(4, 3)),
            ("social_network", self.generate_social_network(30, 4)),
            ("large_sparse", self.generate_random_graph(100, 0.03)),
            ("dense_small", self.generate_random_graph(15, 0.4))
        ]
        
        for name, graph in test_cases:
            # Ajouter du bruit pour rendre plus réaliste
            noisy_graph = self.add_noise_and_dead_ends(graph)
            
            filename = f"{output_dir}/test_{name}.txt"
            self.save_graph(noisy_graph, filename)
            
            print(f"Généré: {filename}")
            print(f"  - Nœuds: {len(noisy_graph)}")
            print(f"  - Arêtes: {sum(len(links) for links in noisy_graph.values())}")
            print(f"  - Culs-de-sac: {sum(1 for links in noisy_graph.values() if not links)}")
            print()

def main():
    parser = argparse.ArgumentParser(description="Générateur de graphes de test pour PageRank")
    parser.add_argument("--type", choices=["random", "scale_free", "hub_spoke", "hierarchical", "social", "suite"],
                       default="suite", help="Type de graphe à générer")
    parser.add_argument("--nodes", type=int, default=20, help="Nombre de nœuds")
    parser.add_argument("--output", default="test_graph.txt", help="Fichier de sortie")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    
    args = parser.parse_args()
    
    generator = WebGraphGenerator(args.seed)
    
    if args.type == "suite":
        print("Génération d'une suite complète de graphes de test...")
        generator.generate_test_suite()
    else:
        print(f"Génération d'un graphe {args.type} avec {args.nodes} nœuds...")
        
        if args.type == "random":
            graph = generator.generate_random_graph(args.nodes, 0.1)
        elif args.type == "scale_free":
            graph = generator.generate_scale_free_graph(args.nodes, 2)
        elif args.type == "hub_spoke":
            graph = generator.generate_hub_and_spoke(max(1, args.nodes // 10), 8)
        elif args.type == "hierarchical":
            graph = generator.generate_hierarchical_graph(3, 3)
        elif args.type == "social":
            graph = generator.generate_social_network(args.nodes, 4)
        
        # Ajouter du bruit
        graph = generator.add_noise_and_dead_ends(graph)
        
        generator.save_graph(graph, args.output)
        print(f"Graphe sauvegardé dans {args.output}")
        print(f"Nœuds: {len(graph)}, Arêtes: {sum(len(links) for links in graph.values())}")

if __name__ == "__main__":
    main()