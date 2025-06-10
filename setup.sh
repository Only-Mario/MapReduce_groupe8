#!/bin/bash
# setup.sh

echo "🔧 Installation de PageRank MapReduce"
echo "------------------------------------"

# Vérifier que Python 3 est installé
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

# Installer les dépendances
echo "📦 Installation des dépendances Python..."
python3 -m pip install -r requirements.txt

# Rendre les scripts exécutables
chmod +x mapper.py reducer.py run_pagerank.sh

echo "✅ Installation terminée"
echo "Pour lancer l'interface : streamlit run app.py"