import streamlit as st
import subprocess
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

st.set_page_config(page_title="Analyseur de Graphes Web - PageRank MapReduce", layout="wide")

st.title("🌐 Analyseur de Graphes Web avec PageRank MapReduce")
st.markdown("**Simulation d'un moteur de recherche utilisant MapReduce et PageRank**")

# Sidebar pour la configuration
st.sidebar.header("⚙️ Configuration")
damping_factor = st.sidebar.slider("Facteur d'amortissement (d)", 0.1, 0.9, 0.85, 0.05)
max_iterations = st.sidebar.slider("Nombre d'itérations max", 5, 50, 20, 5)
convergence_threshold = st.sidebar.number_input("Seuil de convergence", 0.0001, 0.01, 0.001, format="%.4f")

# Colonnes principales
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Données d'entrée")
    
    # Options pour les données d'entrée
    input_option = st.radio("Source des données", ["Fichier exemple", "Upload fichier", "Saisie manuelle"])
    
    if input_option == "Fichier exemple":
        # Graphe d'exemple plus complexe
        example_graphs = {
            "Graphe simple": "A B C\nB C\nC A\nD A B C\nE",
            "Réseau social": "Alice Bob Charlie\nBob Charlie Diana\nCharlie Alice\nDiana Alice Bob\nEve Bob\nFrank Alice Diana",
            "Site web e-commerce": "Homepage Products Cart\nProducts Item1 Item2 Item3\nItem1 Cart Reviews\nItem2 Cart\nItem3 Homepage\nCart Checkout\nReviews Products\nCheckout Homepage"
        }
        
        selected_graph = st.selectbox("Choisir un graphe", list(example_graphs.keys()))
        graph_content = example_graphs[selected_graph]
        st.text_area("Contenu du graphe", graph_content, height=150)
        
        # Sauvegarder le graphe sélectionné
        with open("input.txt", "w") as f:
            f.write(graph_content)
    
    elif input_option == "Upload fichier":
        uploaded_file = st.file_uploader("Choisir un fichier", type=['txt'])
        if uploaded_file:
            graph_content = uploaded_file.read().decode()
            st.text_area("Contenu du fichier", graph_content, height=150)
            with open("input.txt", "w") as f:
                f.write(graph_content)
    
    else:  # Saisie manuelle
        graph_content = st.text_area("Saisir le graphe (format: noeud liens)", height=150)
        if graph_content:
            with open("input.txt", "w") as f:
                f.write(graph_content)

with col2:
    st.header("🎯 Actions")
    
    col2a, col2b = st.columns(2)
    
    with col2a:
        if st.button("🚀 Lancer PageRank", type="primary"):
            if Path("input.txt").exists():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Mise à jour du script bash avec les paramètres
                bash_script = f"""#!/bin/bash
INPUT="input.txt"
MAX_ITER={max_iterations}
THRESHOLD={convergence_threshold}
DAMPING={damping_factor}

cp "$INPUT" iteration0.txt
echo "Démarrage du calcul PageRank..."

for i in $(seq 1 $MAX_ITER)
do
    echo "Itération $i/$MAX_ITER"
    cat iteration$((i-1)).txt | python3 mapper.py | sort | python3 reducer.py $DAMPING > iteration$i.txt
    
    # Vérification de convergence (simplifiée)
    if [ $i -gt 1 ]; then
        DIFF=$(python3 -c "
import sys
try:
    with open('iteration$((i-1)).txt') as f1, open('iteration$i.txt') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        total_diff = 0
        for l1, l2 in zip(lines1, lines2):
            try:
                rank1 = float(l1.split()[1])
                rank2 = float(l2.split()[1])
                total_diff += abs(rank1 - rank2)
            except:
                pass
        print(total_diff)
except:
    print(1.0)
")
        if (( $(echo "$DIFF < $THRESHOLD" | bc -l) )); then
            echo "Convergence atteinte à l'itération $i"
            break
        fi
    fi
done

echo "Calcul terminé."
"""
                
                with open("run_pagerank.sh", "w") as f:
                    f.write(bash_script)
                
                # Exécution
                process = subprocess.Popen(["bash", "run_pagerank.sh"], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE, 
                                         universal_newlines=True)
                
                for i in range(max_iterations):
                    time.sleep(0.1)
                    progress_bar.progress((i + 1) / max_iterations)
                    status_text.text(f"Itération {i + 1}/{max_iterations}")
                
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    st.success("✅ Calcul PageRank terminé!")
                else:
                    st.error(f"❌ Erreur: {stderr}")
            else:
                st.error("❌ Aucun fichier d'entrée trouvé")
    
    with col2b:
        if st.button("📈 Analyser Graphe"):
            if Path("input.txt").exists():
                with st.spinner("Analyse en cours..."):
                    # Analyse de base du graphe
                    G = nx.DiGraph()
                    with open("input.txt") as f:
                        for line in f:
                            nodes = line.strip().split()
                            if len(nodes) > 1:
                                for target in nodes[1:]:
                                    G.add_edge(nodes[0], target)
                            elif len(nodes) == 1:
                                G.add_node(nodes[0])
                    
                    st.write("### 📊 Statistiques du graphe")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Nœuds", G.number_of_nodes())
                    with stats_col2:
                        st.metric("Arêtes", G.number_of_edges())
                    with stats_col3:
                        density = nx.density(G)
                        st.metric("Densité", f"{density:.3f}")

# Section des résultats
st.header("📋 Résultats")

tabs = st.tabs(["PageRank Final", "Évolution", "Visualisation", "Métriques Avancées"])

with tabs[0]:
    # Affichage du PageRank final
    final_files = [f for f in Path(".").glob("iteration*.txt") if f.name != "iteration0.txt"]
    if final_files:
        latest_file = max(final_files, key=lambda x: int(x.stem.replace("iteration", "")))
        
        st.subheader(f"🏆 Résultats finaux ({latest_file.name})")
        
        with open(latest_file) as f:
            content = f.read()
            
        # Parse et trie les résultats
        results = []
        for line in content.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        results.append((parts[0], float(parts[1])))
                    except ValueError:
                        pass
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Affichage sous forme de tableau
        if results:
            df = pd.DataFrame(results, columns=['Page', 'PageRank'])
            df['Rang'] = range(1, len(df) + 1)
            df = df[['Rang', 'Page', 'PageRank']]
            
            st.dataframe(df, use_container_width=True)
            
            # Graphique en barres
            fig, ax = plt.subplots(figsize=(10, 6))
            pages = [r[0] for r in results[:10]]  # Top 10
            ranks = [r[1] for r in results[:10]]
            
            bars = ax.bar(pages, ranks, color='skyblue', alpha=0.7)
            ax.set_xlabel('Pages')
            ax.set_ylabel('Score PageRank')
            ax.set_title('Top 10 - Scores PageRank')
            ax.tick_params(axis='x', rotation=45)
            
            # Ajout des valeurs sur les barres
            for bar, rank in zip(bars, ranks):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{rank:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)

with tabs[1]:
    # Évolution du PageRank
    st.subheader("📈 Évolution des scores PageRank")
    
    iteration_files = sorted([f for f in Path(".").glob("iteration*.txt")], 
                           key=lambda x: int(x.stem.replace("iteration", "")))
    
    if len(iteration_files) > 1:
        evolution_data = {}
        
        for file in iteration_files:
            iteration_num = int(file.stem.replace("iteration", ""))
            with open(file) as f:
                for line in f:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                page = parts[0]
                                rank = float(parts[1])
                                if page not in evolution_data:
                                    evolution_data[page] = {}
                                evolution_data[page][iteration_num] = rank
                            except ValueError:
                                pass
        
        if evolution_data:
            # Graphique d'évolution
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for page, iterations in evolution_data.items():
                x = sorted(iterations.keys())
                y = [iterations[i] for i in x]
                ax.plot(x, y, marker='o', label=page, linewidth=2)
            
            ax.set_xlabel('Itération')
            ax.set_ylabel('Score PageRank')
            ax.set_title('Évolution des scores PageRank par itération')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tableau de convergence
            st.subheader("📊 Analyse de convergence")
            convergence_df = pd.DataFrame(evolution_data).T
            convergence_df = convergence_df.fillna(0)
            st.dataframe(convergence_df, use_container_width=True)

with tabs[2]:
    # Visualisation du graphe
    st.subheader("🕸️ Visualisation du graphe")
    
    if Path("input.txt").exists():
        G = nx.DiGraph()
        with open("input.txt") as f:
            for line in f:
                nodes = line.strip().split()
                if len(nodes) > 1:
                    for target in nodes[1:]:
                        G.add_edge(nodes[0], target)
                elif len(nodes) == 1:
                    G.add_node(nodes[0])
        
        if G.number_of_nodes() > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Graphe original
            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', 
                   node_size=1500, font_size=10, font_weight='bold',
                   arrows=True, arrowsize=20)
            ax1.set_title("Structure du graphe")
            
            # Graphe avec PageRank (si disponible)
            final_files = [f for f in Path(".").glob("iteration*.txt") if f.name != "iteration0.txt"]
            if final_files:
                latest_file = max(final_files, key=lambda x: int(x.stem.replace("iteration", "")))
                
                pagerank_scores = {}
                with open(latest_file) as f:
                    for line in f:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    pagerank_scores[parts[0]] = float(parts[1])
                                except ValueError:
                                    pass
                
                if pagerank_scores:
                    # Normaliser les tailles des nœuds
                    max_score = max(pagerank_scores.values())
                    min_score = min(pagerank_scores.values())
                    
                    node_sizes = []
                    node_colors = []
                    for node in G.nodes():
                        score = pagerank_scores.get(node, 0)
                        # Taille entre 500 et 3000
                        size = 500 + (score - min_score) / (max_score - min_score) * 2500
                        node_sizes.append(size)
                        # Couleur basée sur le score
                        node_colors.append(score)
                    
                    nx.draw(G, pos, ax=ax2, with_labels=True, 
                           node_size=node_sizes, node_color=node_colors,
                           cmap='Reds', font_size=8, font_weight='bold',
                           arrows=True, arrowsize=20)
                    ax2.set_title("Graphe avec scores PageRank")
                    
                    # Ajouter une colorbar
                    sm = plt.cm.ScalarMappable(cmap='Reds', 
                                             norm=plt.Normalize(vmin=min_score, vmax=max_score))
                    sm.set_array([])
                    plt.colorbar(sm, ax=ax2, label='Score PageRank')
            
            plt.tight_layout()
            st.pyplot(fig)

with tabs[3]:
    # Métriques avancées
    st.subheader("🔍 Métriques avancées du graphe")
    
    if Path("input.txt").exists():
        G = nx.DiGraph()
        with open("input.txt") as f:
            for line in f:
                nodes = line.strip().split()
                if len(nodes) > 1:
                    for target in nodes[1:]:
                        G.add_edge(nodes[0], target)
                elif len(nodes) == 1:
                    G.add_node(nodes[0])
        
        if G.number_of_nodes() > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Centralités")
                
                # Calcul des centralités
                try:
                    in_centrality = nx.in_degree_centrality(G)
                    out_centrality = nx.out_degree_centrality(G)
                    closeness_centrality = nx.closeness_centrality(G)
                    
                    centrality_df = pd.DataFrame({
                        'Node': list(G.nodes()),
                        'In-Degree': [in_centrality[node] for node in G.nodes()],
                        'Out-Degree': [out_centrality[node] for node in G.nodes()],
                        'Closeness': [closeness_centrality[node] for node in G.nodes()]
                    })
                    
                    st.dataframe(centrality_df, use_container_width=True)
                except:
                    st.write("Impossible de calculer les centralités pour ce graphe")
            
            with col2:
                st.write("### 🔗 Analyse des liens")
                
                # Distribution des degrés
                in_degrees = [G.in_degree(node) for node in G.nodes()]
                out_degrees = [G.out_degree(node) for node in G.nodes()]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
                
                ax1.hist(in_degrees, bins=max(1, len(set(in_degrees))), alpha=0.7, color='blue')
                ax1.set_title('Distribution des degrés entrants')
                ax1.set_xlabel('Degré entrant')
                ax1.set_ylabel('Fréquence')
                
                ax2.hist(out_degrees, bins=max(1, len(set(out_degrees))), alpha=0.7, color='red')
                ax2.set_title('Distribution des degrés sortants')
                ax2.set_xlabel('Degré sortant')
                ax2.set_ylabel('Fréquence')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistiques
                st.write("### 📈 Statistiques")
                stats = {
                    "Degré entrant moyen": np.mean(in_degrees),
                    "Degré sortant moyen": np.mean(out_degrees),
                    "Nœuds isolés": len([n for n in G.nodes() if G.degree(n) == 0]),
                    "Composantes fortement connexes": nx.number_strongly_connected_components(G)
                }
                
                for key, value in stats.items():
                    st.metric(key, f"{value:.2f}" if isinstance(value, float) else value)

# Footer
st.markdown("---")
st.markdown("💡 **Conseils d'utilisation:**")
st.markdown("""
- **Format d'entrée:** Chaque ligne représente un nœud et ses liens sortants
- **Paramètres:** Ajustez le facteur d'amortissement (0.85 par défaut) et le nombre d'itérations
- **Convergence:** L'algorithme s'arrête quand les scores se stabilisent
- **Interprétation:** Un score PageRank élevé indique une page importante dans le réseau
""")

st.markdown("🔧 **Applications:** Moteurs de recherche, analyse de réseaux sociaux, systèmes de recommandation")