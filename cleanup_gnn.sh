#!/bin/bash
# SpinLab GNN Cleanup Script
# This script removes all GNN-related files from SpinLab

SPINLAB_DIR="/Users/akramibrahim/SpinLab"

echo "🧹 Cleaning up GNN files from SpinLab..."

# Check if SpinLab directory exists
if [ ! -d "$SPINLAB_DIR" ]; then
    echo "❌ SpinLab directory not found at $SPINLAB_DIR"
    exit 1
fi

cd "$SPINLAB_DIR"

# Remove GNN files
echo "Removing GNN Hamiltonian..."
if [ -f "spinlab/core/gnn_hamiltonian.py" ]; then
    rm spinlab/core/gnn_hamiltonian.py
    echo "✅ Removed spinlab/core/gnn_hamiltonian.py"
else
    echo "⚠️  spinlab/core/gnn_hamiltonian.py not found"
fi

echo "Removing GNN example..."
if [ -f "examples/gnn_integration_example.py" ]; then
    rm examples/gnn_integration_example.py
    echo "✅ Removed examples/gnn_integration_example.py"
else
    echo "⚠️  examples/gnn_integration_example.py not found"
fi

# Verify cleanup
echo ""
echo "🔍 Verifying cleanup..."
remaining_gnn_files=$(find . -name "*gnn*" -type f 2>/dev/null)
if [ -z "$remaining_gnn_files" ]; then
    echo "✅ All GNN files successfully removed!"
    echo "🎉 SpinLab is now completely free of GNN dependencies"
else
    echo "⚠️  Some GNN files may still remain:"
    echo "$remaining_gnn_files"
fi

echo ""
echo "🚀 SpinLab is ready with cluster expansion functionality!"
echo "   Use: from spinlab import ClusterExpansionBuilder"