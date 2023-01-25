# multilayer-perceptron

Traduction de [Coding A Neural Network From Scratch in NumPy](https://towardsdatascience.com/coding-a-neural-network-from-scratch-in-numpy-31f04e4d605)

### Introduction
Dans cet article, je vais vous expliquer comment développer un réseau de neurones artificiels à partir de zéro en utilisant NumPy. L'architecture de ce modèle est la plus basique de tous les réseaux neuronaux artificiels : un réseau simple de type feed-forward. Je vais également montrer l'équivalent Keras de ce modèle, car j'ai essayé de rendre mon implémentation " Keras-esque ". Bien que l'architecture feed-forward soit basique par rapport à d'autres réseaux neuronaux tels que les transformateurs, ces concepts de base peuvent être extrapolés pour construire des ANN plus complexes. Ces sujets sont intrinsèquement techniques. Pour un article plus conceptuel sur l'IA, veuillez consulter mon autre article, [Demystifying Artificial Intelligence.](https://medium.com/geekculture/demystifying-artificial-intelligence-bdd9a117d4a6)

### Table des matières
* Aperçu de l'architecture
  * Forward Pass (Passe avant)
  * Backward Pass (Passe arrière)
* Implémentation de NumPy
  * Data
  * Construction des couches (Layers)
  * Construction du réseau (Network)
  * **Réseau**: Forward Pass
  * **Layers**: Forward Pass
  * Effectuer un *Forward Pass* / Contrôle
  * **Réseau**: Backward Pass
  * **Layers**: Backward Pass
  * Effectuer un *Backward Pass* / Contrôle
  * Entraîner un *model*
* Conclusion

### Conventions
* X = inputs (Entrées)
* y = labels (Etiquettes)
* W = weights (Poids)
* b = bias (biais)
* Z = produit scalaire de X et W plus b (dot)
* A = activation(Z)
* k = nombre de classes
* Les lettres minuscules désignent les vecteurs, les lettres majuscules désignent les matrices.

## Architecture
### Forward Pass (Passe avant)

