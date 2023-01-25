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

*Le produit scalaire (dot product)*

Tout d'abord, nous calculons le produit scalaire de nos entrées et de nos pondérations et nous ajoutons un terme de biais.

### $$Z = \sum_i^n (X_i * W_i) + b$$

Deuxièmement, nous faisons passer la somme pondérée obtenue à la première étape par une fonction d'activation.

Ces deux opérations sont basées sur des éléments et sont simples. Par conséquent, je n'entrerai pas dans les détails. Ces calculs ont lieu dans chaque neurone de chaque couche cachée.

### $$A = \sigma (Z) = \sigma (\sum_i^n (X_i * W_i) + b)$$

*Fonction d'Activation*

Dans mon implémentation, nous utilisons l'activation ReLU dans les couches cachées car elle est facile à différencier, et l'activation Softmax dans la couche de sortie (plus de détails ci-dessous). Dans les prochaines versions, je vais le développer pour qu'il soit plus robuste et qu'il permette toutes ces fonctions d'activation.
### Fonctions d'activation couramment utilisées :

* Sigmoïd  $$f(z) = \frac{1}{1 + e^{-z}}$$
* Tanh $$f(x) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
* ReLu  $$f(z) = (z, z > 0, else \~ 0)$$

## Backward Pass (Passe Arrière)
*La fonction de perte (Loss Function)*

Nous commençons par calculer la perte, également appelée erreur. Il s'agit d'une mesure du degré d'erreur du modèle.

La perte est une fonction différentielle avec laquelle nous allons entraîner le modèle à minimiser. Selon la tâche que vous essayez d'effectuer, vous pouvez choisir une fonction de perte différente. Dans mon implémentation, nous utilisons la perte d'entropie croisée (*cross-entropy loss*) car il s'agit d'une tâche de classification multiple, comme illustré ci-dessous. Pour une tâche de classification binaire, vous pouvez utiliser la perte d'entropie croisée binaire (*binary cross-entropy loss*), pour une tâche de régression, l'erreur quadratique moyenne.

* Cross-Entropy Loss  $$L = - \frac{1}{m} \sum_{i=1}^m y_i \bullet log(\hat{y}_i)$$

Cela m'a causé une certaine confusion, j'aimerais donc développer ce qui se passe ici. La formule ci-dessus implique que les étiquettes sont codées en une fois. Keras s'attend à ce que les étiquettes soient codées à un coup, mais ma mise en œuvre ne le fait pas. Voici un exemple de calcul de la perte d'entropie croisée, et un exemple de la raison pour laquelle **il n'est pas nécessaire de coder les étiquettes en une seule fois.**

Étant donné les données suivantes d'un seul échantillon, les étiquettes codées à un coup (y) et la prédiction de notre modèle (yhat), nous calculons la perte d'entropie croisée (cross-entropy loss).

### $y = [1, 0, 0]$
### $ŷ = [3.01929735e-07, 7.83961013e-09, 9.99999690e-01]$

```python
>>> loss = (-np.log(yhat[0]) * ý[0]) + (-np.log(yhat[1] * y[1]) + (-np.log(yhat[2]) * y[2])
>>> loss
15.013071512205286
>>>
```
Comme vous pouvez le voir, la classe correcte à cet échantillon était zéro, indiquée par un 1 dans l'indice zéro du tableau y. Nous multiplions le log négatif de notre probabilité de sortie, par l'étiquette correspondante pour cette classe, et nous additionnons toutes les classes.

Vous l'avez peut-être déjà remarqué, mais outre l'indice zéro, nous obtenons zéro, car tout ce qui est multiplié par zéro est zéro. Ce que cela donne, c'est simplement le log négatif de notre probabilité à l'indice correspondant pour la classe correcte. Ici, la classe correcte était zéro, donc nous prenons le log négatif de nos probabilités à l'indice zéro.
```python
>>> -np.log(3.01919735e-07)
15.013071512205286
>>>
```
La perte totale est la moyenne de tous les échantillons, désignée par m dans l'équation. Pour obtenir ce chiffre, il faut répéter le calcul ci-dessus pour chaque échantillon, calculer la somme et la diviser par le nombre total d'échantillons.
