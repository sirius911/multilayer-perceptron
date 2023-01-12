# multilayer-perceptron

[Make your own machine learning library.](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65)

Nous allons parcourir les mathématiques de l'apprentissage automatique et coder à partir de zéro, en Python, une petite bibliothèque pour construire des réseaux de neurones avec une variété de couches (Fully Connected, Convolutional, etc.). À terme, nous serons en mesure de créer des réseaux de manière modulaire :

Je suppose que vous avez déjà quelques connaissances sur les réseaux neuronaux. Le but ici n'est pas d'expliquer pourquoi nous faisons ces modèles, mais de montrer comment faire une implémentation correcte.

# Couche par couche (Layer)
Nous devons garder à l'esprit la situation dans son ensemble :

1. Nous introduisons des données d'entrée dans le réseau neuronal.
2. Les données circulent de couche en couche jusqu'à ce que nous obtenions la sortie.
3. Une fois que nous avons la sortie, nous pouvons calculer l'erreur qui est un scalaire.
4. Enfin, nous pouvons ajuster un paramètre donné (poids ou biais) en soustrayant la dérivée de l'erreur par rapport au paramètre lui-même.
5. Nous itérons à travers ce processus.

L'étape la plus importante est la quatrième. Nous voulons pouvoir avoir autant de couches que nous le souhaitons, et de n'importe quel type.
Mais si nous modifions/ajoutons/enlevons une couche du réseau, la sortie du réseau va changer, ce qui va changer l'erreur, ce qui va changer la dérivée de l'erreur par rapport aux paramètres. Nous devons être capables de calculer les dérivées quelle que soit l'architecture du réseau, quelles que soient les fonctions d'activation, quelle que soit la perte utilisée.

Pour ce faire, nous devons mettre en œuvre chaque couche séparément.

### Ce que chaque couche devrait implémenter
Toutes les couches que nous pouvons créer, **fully connected** (entièrement connectées), **convolutional** (convolutionnelles), maxpooling, dropout, etc., ont au moins deux choses en commun : des données d'entrée(**input**) et de sortie(**output**).

### $$X \to \fbox{Couche} \to Y$$

### Propagation vers l'avant (Forward propagation)
Nous pouvons déjà souligner un point important, à savoir que la sortie d'une couche est l'entrée de la couche suivante.

### $$X \to \fbox{Couche\~1} \to \fbox{Couche\~2} \to \fbox{Couche\~3} \to Y\~E$$

C'est ce qu'on appelle la propagation vers l'avant(**Forward propagation**).
Essentiellement, nous donnons les données d'entrée à la première couche, puis la sortie de chaque couche devient l'entrée de la couche suivante jusqu'à ce que nous atteignions la fin du réseau.
En comparant le résultat du réseau (Y) avec la sortie souhaitée (disons Y*), nous pouvons calculer une erreur E.
Le but est de **minimiser** cette erreur en modifiant les paramètres du réseau. C'est la propagation à rebours (backpropagation).

### Descente de gradient
Ceci est un rappel rapide, si vous avez besoin d'en savoir plus sur la descente de gradient, il y a des tonnes de ressources sur internet.

Fondamentalement, nous voulons changer un paramètre du réseau (appelé w) de sorte que l'erreur totale E diminue. Il existe une façon intelligente de le faire (pas au hasard) qui est la suivante :
$$w\leftarrow w - \alpha\frac{\partial E}{\partial w}$$
Où **$\alpha$** est un paramètre dans l'intervalle [0,1] que nous fixons et qui est appelé le taux d'apprentissage(**learning rate**).
Quoi qu'il en soit, l'élément important ici est **$\frac{\partial E}{\partial w}$** (la dérivée de E par rapport à w).
**Nous devons être capables de trouver la valeur de cette expression pour n'importe quel paramètre du réseau, quelle que soit son architecture.**

### Propagation vers l'arrière (Backward propagation)
Supposons que nous donnions à une couche **la dérivée de l'erreur par rapport à sa sortie** $(\frac{\partial E}{\partial Y})$, alors elle doit être capable de fournir **la dérivée de l'erreur par rapport à son entrée** $(\frac{\partial E}{\partial X})$.

### $$\frac{\partial E}{\partial X} \leftarrow \fbox{Couche} \leftarrow \frac{\partial E}{\partial Y}$$
Rappelez-vous que E est un **scalaire** (un nombre) et que X et Y sont des **matrices**.

### $$\frac{\partial E}{\partial X} = [\frac{\partial E}{\partial x_1} \~ \frac{\partial E}{\partial x_2} \~ ... \~ \frac{\partial E}{\partial x_i}]$$

### $$\frac{\partial E}{\partial Y} = [\frac{\partial E}{\partial y_1} \~ \frac{\partial E}{\partial y_2} \~ ... \~ \frac{\partial E}{\partial y_i}]$$

Oublions $\frac{\partial E}{\partial X}$ pour l'instant. L'astuce ici, est que si nous avons accès à $\frac{\partial E}{\partial Y}$ nous pouvons très facilement calculer $\frac{\partial E}{\partial W}$ (si la couche a des paramètres entraînables) sans rien savoir de l'architecture du réseau ! Nous utilisons simplement la règle de la chaîne :

### $$\frac{\partial E}{\partial w} = \sum_j\frac{\partial E}{\partial y_j}\frac{\partial y_j}{\partial w}$$

L'inconnue est $\frac{\partial y_j}{\partial w}$ qui dépend totalement de la façon dont la couche calcule sa sortie. Donc si chaque couche a accès à $\frac{\partial E}{\partial Y}$, où Y est sa propre sortie, alors nous pouvons mettre à jour nos paramètres !

### Mais pourquoi $\frac{\partial E}{\partial X}$ ?
N'oubliez pas que la sortie d'une couche est l'entrée de la couche suivante. Ce qui signifie que $\frac{\partial E}{\partial X}$ pour une couche est $\frac{\partial E}{\partial Y}$ pour la couche précédente ! Voilà, c'est tout ! C'est juste une façon astucieuse de propager l'erreur ! Encore une fois, nous pouvons utiliser la règle de la chaîne :

### $$\frac{\partial E}{\partial x_i} = \sum_j\frac{\partial E}{\partial y_j}\frac{\partial y_j}{\partial x_i}$$

C'est très important, c'est la *clé* pour comprendre la rétro-propagation ! Après cela, nous serons capables de coder un réseau de neurones convolutifs profonds en un rien de temps !

### Diagramme pour comprendre la rétro-propagation
C'est ce que j'ai décrit précédemment. La couche 3 (layer 3) va mettre à jour ses paramètres en utilisant $\frac{\partial E}{\partial Y}$, et va ensuite transmettre $\frac{\partial E}{\partial H_2}$ à la couche précédente, qui est son propre "∂E/∂Y". La couche 2 (layer 2) va ensuite faire de même, et ainsi de suite.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25301163/211849486-686db912-e2ba-4843-8e82-655ca5e5980b.jpg">
</p>

Cela peut sembler abstrait ici, mais cela deviendra très clair lorsque nous l'appliquerons à un type de couche spécifique. En parlant d'abstrait, c'est le bon moment pour écrire notre première classe python.

### Classe de base abstraite : Layer
La classe abstraite Layer, dont toutes les autres couches hériteront, gère des propriétés simples qui sont une entrée (**input**), une sortie (**output**), et des méthodes avant (**forward**) et arrière (**backward**).

```python
# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
```
background-color:yellow
Comme vous pouvez le voir, il y a un paramètre supplémentaire dans **backward_propagation**, que je n'ai pas mentionné, c'est le **learning_rate**. Ce paramètre devrait être quelque chose comme une politique de mise à jour, ou un optimiseur comme ils l'appellent dans *Keras*, mais pour des raisons de simplicité, nous allons simplement passer un taux d'apprentissage et mettre à jour nos paramètres en utilisant la descente de gradient.

### Couche entièrement connectée (Fully Connected Layer)
Définissons et implémentons maintenant le premier type de couche : la couche entièrement connectée ou couche FC (**F**ully **C**onnected). Les couches FC sont les couches les plus basiques car chaque neurone d'entrée est connecté à chaque neurone de sortie.
<p align="center">
  <img src="https://user-images.githubusercontent.com/25301163/212079310-d5f62246-3c79-41af-a023-e15c25d0f0e7.jpg">
</p>

### Propagation vers l'avant (Forward Propagation)
La valeur de chaque neurone de sortie peut être calculée comme suit :

### $$y_j = b_j + \sum_i x_iw_{ij}$$

Avec les matrices, nous pouvons calculer cette formule pour chaque neurone de sortie en une seule fois en utilisant un produit scalaire (**dot product**):

<p align="center">
  <img src="https://user-images.githubusercontent.com/25301163/212083330-dc84ea78-2a30-4a57-994d-b64c68c15f1d.jpg">
</p>

## $$Y = XW + B$$

Nous avons fini avec la passe avant (Forward Propagation). Maintenant, faisons la passe arrière (backward) de la couche FC.
*Notez que je n'utilise pas encore de fonction d'activation, c'est parce que nous allons l'implémenter dans une couche séparée !*

### Propagation vers l'arrière (Backward Propagation)
Comme nous l'avons dit, supposons que nous ayons une matrice contenant la dérivée de l'erreur par rapport à la sortie de cette couche $\frac{\partial E}{\partial Y}$. Nous avons besoin de :
1. La dérivée de l'erreur par rapport aux paramètres ( $\frac{\partial E}{\partial W}$, $\frac{\partial E}{\partial B}$ )
2. La dérivée de l'erreur par rapport à l'entrée ( $\frac{\partial E}{\partial X}$ )

Calculons $\frac{\partial E}{\partial W}$. Cette matrice doit être de la même taille que W lui-même : ixj où i est le nombre de neurones d'entrée et j le nombre de neurones de sortie. Nous avons besoin d'un gradient pour chaque poids :

<p align="center">
  <img src="https://user-images.githubusercontent.com/25301163/212091867-9ec4aad0-c618-48b8-a98e-3c3644b97448.jpg">
</p>
En utilisant la règle de la chaîne énoncée précédemment, nous pouvons écrire :

$$
\begin{aligned}
\frac{\partial E}{\partial w_{ij} }&=\frac{\partial E}{\partial y_1}\frac{\partial y_1}{\partial w_{ij} } + \cdots + \frac{\partial E}{\partial y_j}\frac{\partial y_j}{\partial w_{ij}}\\
&= \frac{\partial E}{\partial y_j}x_i
\end{aligned}
$$

Par conséquent,

<p align="center">
  <img src="https://user-images.githubusercontent.com/25301163/212097948-5f2697fe-b9fe-40cc-a123-2f03ec750a54.jpg">
</p>

Voilà, nous avons la première formule pour mettre à jour les poids ! Maintenant, nous allons calculer $\frac{\partial E}{\partial B}.

### $$\frac{\partial E}{\partial B} = [ \frac{\partial E}{\partial b_1} \~ \frac{\partial E}{\partial b_2} + \cdots + \frac{\partial E}{\partial b_j} ]$$

De nouveau, $\frac{\partial E}{\partial B}$ doit être de la même taille que B lui-même, un gradient par biais. Nous pouvons à nouveau utiliser la règle de la chaîne :

$$
\begin{aligned}
\frac{\partial E}{\partial b_j}&=\frac{\partial E}{\partial y_1}\frac{\partial y_1}{\partial b_j} + \cdots + \frac{\partial E}{\partial y_j}\frac{\partial y_j}{\partial b_j}\\
&= \frac{\partial E}{\partial y_j}
\end{aligned}
$$

Et de conclure que,

$$
\begin{aligned}
\frac{\partial E}{\partial B}&=\frac{\partial E}{\partial y_1} \~ \frac{\partial y_1}{\partial y_2} + \cdots + \frac{\partial E}{\partial y_j}\\
&= \frac{\partial E}{\partial Y}
\end{aligned}
$$

Maintenant que nous avons $\frac{\partial E}{\partial W}$ et $\frac{\partial E}{\partial B}$ , il nous reste $\frac{\partial E}{\partial X}$ qui est **très importante** car elle va "agir" comme $\frac{\partial E}{\partial Y}$ pour la couche qui la précède.

### $$\frac{\partial E}{\partial X} = [\frac{\partial E}{\partial x_1} \~ \frac{\partial E}{\partial x_2} \cdots \frac{\partial E}{\partial x_i}]$$

Encore une fois, en utilisant la règle de la chaîne,

$$
\begin{aligned}
\frac{\partial E}{\partial x_i}&=\frac{\partial E}{\partial y_1}\frac{\partial y_1}{\partial x_i} + \cdots + \frac{\partial E}{\partial y_j}\frac{\partial y_j}{\partial x_i}\\
&= \frac{\partial E}{\partial y_1} w_{i1} + \cdots + \frac{\partial E}{\partial y_j} w_{ij}
\end{aligned}
$$

Finally, we can write the whole matrix :

<p align="center">
  <img src="https://user-images.githubusercontent.com/25301163/212103303-797e6c9d-1035-469f-85e9-e3b991fafe33.jpg">
</p>

Voilà, c'est fait ! Nous avons les trois formules dont nous avions besoin pour la couche FC !

$$
\begin{aligned}
\frac{\partial E}{\partial X}&=\frac{\partial E}{\partial Y}W^t\\
\frac{\partial E}{\partial W}&= X^t\frac{\partial E}{\partial Y}\\
\frac{\partial E}{\partial B}&= \frac{\partial E}{\partial Y}
\end{aligned}
$$

### Codage de la couche entièrement connectée (Fully Connected Layer)
Nous pouvons maintenant écrire du code python pour donner vie à ces mathématiques !
```python
from layer import Layer
import numpy as np

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
```

### Couche d'activation (Activation Layer)
Tous les calculs que nous avons faits jusqu'à présent étaient complètement linéaires. Il est impossible d'apprendre quoi que ce soit avec ce type de modèle. Nous devons ajouter de la non-linéarité au modèle en appliquant des fonctions non linéaires à la sortie de certaines couches.

Maintenant, nous devons refaire tout le processus pour ce nouveau type de couche !

Pas d'inquiétude, ça va être beaucoup plus rapide car il n'y a pas de paramètres à apprendre. Nous avons juste besoin de calculer $\frac{\partial E}{\partial X}$.

Nous appellerons $f$ et $f'$ respectivement la fonction d'activation et sa dérivée.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25301163/212106716-e3ad51b7-4c39-4a3d-b94b-16f5e275ff42.jpg">
</p>

### Propagation vers l'avant (Forward propagation)
Comme vous le verrez, c'est assez simple. Pour une entrée X donnée, la sortie est simplement la fonction d'activation appliquée à chaque élément de X . Ce qui signifie que **l'entrée** et la **sortie** ont les **mêmes dimensions**.

$$
\begin{aligned}
Y&=[f(x_1) \cdots f(x_i)]\\
&= f(X)\\
\end{aligned}
$$

### Propagation vers l'arrière (Backward propagation)
Étant donné $\frac{\partial E}{\partial Y}$, nous voulons calculer $\frac{\partial E}{\partial X}$.

$$
\begin{aligned}
\frac{\partial E}{\partial X}&=[\frac{\partial E}{\partial x_1} \cdots \frac{\partial E}{\partial x_i}]\\
&= [\frac{\partial E}{\partial y_1}\frac{\partial y_1}{\partial x_1} \cdots \frac{\partial E}{\partial y_i}\frac{\partial y_i}{\partial x_i}]\\
&= [\frac{\partial E}{\partial y_1}f'(x_1) \cdots \frac{E}{y_i}f'(x_i)]\\
&= [\frac{\partial E}{\partial y_1} \cdots \frac{\partial E}{\partial y_i}] \odot [f'(x_1) \cdots f'(x_i)]\\
&= \frac{\partial E}{\partial Y} \odot f'(X)
\end{aligned}
$$

Attention, nous utilisons ici une multiplication **par éléments** entre les deux matrices (alors que dans les formules ci-dessus, il s'agissait d'un produit scalaire).

### Codage de la couche d'activation (activation Layer)
Le code de la couche d'activation est aussi simple.
```python
from layer import Layer

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
```
Vous pouvez également écrire certaines fonctions d'activation et leurs dérivés dans un fichier séparé. Elles seront utilisées plus tard pour créer une couche d'activation.

```python
import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;
```

### Fonction de perte (Loss Function)
Jusqu'à présent, pour une couche donnée, nous supposions que $\frac{\partial E}{\partial Y}$ était donnée (par la couche suivante). Mais que se passe-t-il pour la dernière couche ? Comment obtient-elle $\frac{\partial E}{\partial Y}$ ? Nous le donnons simplement manuellement, et cela dépend de la façon dont nous définissons l'erreur.

C'est vous qui définissez l'erreur du réseau, qui mesure la qualité ou la faiblesse du réseau pour des données d'entrée données. Il existe de nombreuses façons de définir l'erreur, et l'une des plus connues est appelée **MSE (Mean Squared Error)**.
