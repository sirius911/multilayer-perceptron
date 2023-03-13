# multilayer-perceptron

Traduction et adaptation à partir de [Coding A Neural Network From Scratch in NumPy](https://towardsdatascience.com/coding-a-neural-network-from-scratch-in-numpy-31f04e4d605)

### Introduction
Dans cet article, je vais vous expliquer comment développer un réseau de neurones artificiels à partir de zéro en utilisant NumPy. L'architecture de ce modèle est la plus basique de tous les réseaux neuronaux artificiels : un réseau simple de type feed-forward.

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

*Descente de gradient stochastique (Stochastic Gradient Descent)*

Maintenant que nous avons calculé la perte, il est temps de la minimiser. Nous commençons par calculer le gradient de nos probabilités de sortie par rapport aux paramètres d'entrée, puis nous rétropropageons les gradients vers les paramètres de chaque couche.

À chaque couche, nous effectuons des calculs similaires à ceux du Forward Pass, sauf qu'au lieu de faire les calculs pour seulement Z et A, nous devons exécuter un calcul pour chaque paramètre (dZ, dW, db, dA), comme indiqué ci-dessous.

*Hidden Layer (Couche cachée)

* dZ: $$\partial Z^{[L]} = \partial A^{[L]} * g'(Z^{[L]} )$$
* Weight Gradient:  $$\partial W^{[L]} = A_T^{[L - 1 ]} \bullet \partial Z^{[L]}$$
* Bias Gradient:  $$\partial b^{[L]} = \sum_i^m \partial Z^{[L]\~(i) }$$
* Activation Gradient:  $$\partial A^{[L - 1]} = \partial Z^{[L]} \bullet W^{[L]}$$

Il existe un cas particulier de dZ dans la couche de sortie, car nous utilisons l'activation softmax. Ceci est expliqué en profondeur plus loin dans cet article.

## Implémentation NumPy
### Data

J'utiliserai le jeu de données (Dataset) fourni par le sujet.Il s'agit d'un fichier csv de 32 colonnes, la colonne diagnostic étant l'étiquette (label) elle peut prendre la valeur M ou B (pour **M**alin ou **B**énin).
Les caractéristiques de l'ensemble de données décrivent les caractéristiques d'un noyau cellulaire d'une masse mammaire extraite par aspiration à aiguille fine.(pour des informations plus détaillées, cliquez [ici](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)).

A noté que l'on a 569 lignes contenant 32 attributs: (ID, Diagnostic (M/B), 30 caractéristiques d'entrée [features] sous forme de réels).
```
Les colonnes de ce dataset sont les suivantes:

ID: Identifiant unique pour chaque patient
Diagnosis: Résultat du diagnostic (M = maligne, B = bénigne)
Radius Mean: Moyenne des distances depuis le centre jusqu'au périmètre
Texture Mean: Déviation standard des valeurs de la couleur de la zone adjacente aux pixels
Perimeter Mean: Moyenne de la taille de la tumeur
Area Mean: Moyenne de la surface de la tumeur
Smoothness Mean: Moyenne des variations locales de la longueur des rayons
Compactness Mean: Moyenne de la proximité de la zone de la tumeur au centre
Concavity Mean: Moyenne de la sévérité des parties concaves de la contour
Concave Points Mean: Moyenne pour le nombre de points concaves de la contour
Symmetry Mean: Moyenne pour la symétrie
Fractal Dimension Mean: Moyenne pour la mesure de la complexité
Radius SE: Écart-type des distances depuis le centre jusqu'au périmètre
Texture SE: Écart-type des valeurs de la couleur de la zone adjacente aux pixels
Perimeter SE: Écart-type de la taille de la tumeur
Area SE: Écart-type de la surface de la tumeur
Smoothness SE: Écart-type des variations locales de la longueur des rayons
Compactness SE: Écart-type de la proximité de la zone de la tumeur au centre
Concavity SE: Écart-type de la sévérité des parties concaves de la contour
Concave Points SE: Écart-type pour le nombre de points concaves de la contour
Symmetry SE: Écart-type pour la symétrie
Fractal Dimension SE: Écart-type pour la mesure de la complexité
Radius Worst: "Pire" (le plus grand) distance depuis le centre jusqu'au périmètre
Texture Worst: "Pire" (la plus grande) déviation standard des valeurs de la couleur de la zone adjacente aux pixels
Perimeter Worst: "Pire" (la plus grande) taille de la tumeur
Area Worst: "Pire" (la plus grande) surface de la tumeur
Smoothness Worst: "Pire" (la plus grande) variation locale de la longueur des rayons
Compactness Worst: "Pire" (la plus grande) proximité de la zone de la tumeur au centre
Concavity Worst: "Pire" (la plus grande) sévérité des parties concaves de la contour
Concave Points Worst: "Pire" (le plus grand) nombre de points concaves de la contour
Symmetry Worst: "Pire" (la plus grande) symétrie
Fractal Dimension Worst: "Pire" (la plus grande) mesure de la complexité
```
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_data(path):
    data = pd.read_csv(path, header=None)
    X = np.array(data[data.columns[2:]].values)
    X = normalize(X)
    y = np.array(LabelEncoder().fit_transform(target).reshape(-1, 1))
    return np.array(X), np.array(y)

X, y = get_data("data.csv")
```
Les données doivent être Normalisées (par exemple avec un minmax) et les labels (y) transformés en (0 ou 1)

voici quelques infos sur le dataset après traitement: 
```
Shape of Xs = (569, 30) and shape of y = (569,)
Nb of datas : 569
        Nb features : 30
Xs[:3] = [[0.52103744 0.0226581  0.54598853 0.36373277 0.59375282 0.7920373
  0.70313964 0.73111332 0.68636364 0.60551811 0.35614702 0.12046941
  0.3690336  0.27381126 0.15929565 0.35139844 0.13568182 0.30062512
  0.31164518 0.18304244 0.62077552 0.14152452 0.66831017 0.45069799
  0.60113584 0.61929156 0.56861022 0.91202749 0.59846245 0.41886396]
 [0.64314449 0.27257355 0.61578329 0.50159067 0.28987993 0.18176799
  0.20360825 0.34875746 0.37979798 0.14132266 0.15643672 0.08258929
  0.12444047 0.12565979 0.11938675 0.08132304 0.0469697  0.25383595
  0.08453875 0.0911101  0.60690146 0.30357143 0.53981772 0.43521431
  0.34755332 0.15456336 0.19297125 0.63917526 0.23358959 0.22287813]
 [0.60149557 0.3902604  0.59574321 0.44941676 0.51430893 0.4310165
  0.46251172 0.63568588 0.50959596 0.21124684 0.22962158 0.09430251
  0.18037035 0.16292179 0.15083115 0.2839547  0.09676768 0.38984656
  0.20569032 0.12700551 0.55638563 0.36007463 0.50844166 0.37450845
  0.48358978 0.38537513 0.35974441 0.83505155 0.40370589 0.21343303]]
y[:3] = [1 1 1]
```
### Contruction des Couches (Layers)
```python
import numpy as np

class DenseLayer:
    def __init__(self, neurons, act_name='relu'):
        self.neurons = neurons
        self.act_name = act_name
        
    def forward(self, inputs, weights, bias, activation):
        """
        Single Layer Forward Propagation
        """
        raise NotImplementedError
    
    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        """
        Single Layer Backward Propagation
        """
        raise NotImplementedError
```

### Contruction du réseau (Network)

```python
class Network:
    def __init__(self):
        self.network = [] ## layers
        self.architecture = [] ## mapping input neurons --> output neurons
        
    def add(self, layer):
        """
        Add layers to the network
        """
        self.network.append(layer)
            
    def _compile(self, data):
        """
        Initialize model architecture
        """
        raise NotImplementedError
        
    def _forwardprop(self, data):
        """
        Performs one full forward pass through network
        """
        raise NotImplementedError
    
    def _backprop(self, predicted, actual):
        """
        Performs one full backward pass through network
        """
        raise NotImplementedError
            
    def _update(self, lr=0.01):
        """
        Update the model parameters --> lr * gradient
        """
        raise NotImplementedError
    
    def _get_accuracy(self, predicted, actual):
        """
        Calculate accuracy after each iteration
        """
        raise NotImplementedError
    
    def _calculate_loss(self, predicted, actual):
        """
        Calculate cross-entropy loss after each iteration
        """
        raise NotImplementedError
    
    def train(self, X_train, y_train, epochs):
        """
        Train the model using SGD
        """
        raise NotImplementedError
```

## Réseau : Forward Pass
### Architecture
Commençons par initialiser dynamiquement l'architecture du réseau. Cela signifie que nous pouvons initialiser notre architecture de réseau pour un nombre arbitraire de couches et de neurones.
```python
def _compile(self, data):
    """
    Initialize model architecture
    """
    for idx, layer in enumerate(self.network):
            self.architecture.append({'input_dim':data.shape[1], 
                                    'output_dim':self.network[idx].neurons,
                                    'activation':layer.act_name})
            layer.compile(data.shape[1])
            input_shape = self.network[idx].neurons
        return self
```
Nous commençons par créer une matrice qui fait correspondre notre nombre de caractéristiques(features) au nombre de neurones de la couche d'entrée. A partir de là, c'est assez simple - la dimension d'entrée d'une nouvelle couche est le nombre de neurones de la couche précédente, la dimension de sortie est le nombre de neurones de la couche actuelle. On compile ensuite les paramètres pour chaque couche.

```
model = Network()
model.add(DenseLayer(6, 'relu'))
model.add(DenseLayer(8, 'relu'))
model.add(DenseLayer(10, 'relu'))
model.add(DenseLayer(2, 'softmax'))

model._compile(X)

print(model.architecture)

Out -->

[{'input_dim': 4, 'output_dim': 6, 'activation': 'relu'},
 {'input_dim': 6, 'output_dim': 8, 'activation': 'relu'},
 {'input_dim': 8, 'output_dim': 10, 'activation': 'relu'},
 {'input_dim': 10, 'output_dim': 2, 'activation': 'softmax'}]
```

### Paramètres
Maintenant que nous avons créé un réseau, nous devons à nouveau initialiser dynamiquement nos paramètres d'apprentissage (W, b), pour un nombre arbitraire de couches/neurones, dans la classe DenseLayer

```python
def compile(self, input_dim):
        self.input_dim = input_dim
        self.activation = act_funct[self.act_name]
        self.activation_der = der_funct[self.act_name]
        np.random.seed(99)
        self.W = np.random.uniform(low=-1,
                                    high=1,
                                    size=(self.neurons, input_dim))
        self.b = np.zeros((1, self.neurons))
```
Comme vous pouvez le voir, nous créons une matrice de poids (**W**) à chaque couche.

Cette matrice contient un vecteur pour chaque neurone, et une dimension pour chaque caractéristique d'entrée.

Il y a un vecteur de biais (**b**)avec une dimension pour chaque neurone dans une couche.

Remarquez également que nous définissons un *np.random.seed()*, pour obtenir des résultats cohérents à chaque fois. Essayez de commenter cette ligne de code pour voir comment elle affecte vos résultats.

```
model = Network()
model.add(DenseLayer(6))
model.add(DenseLayer(8))
model.add(DenseLayer(10))
model.add(DenseLayer(3))
model._compile(X)
print(model.network[0].W.shape, model.network[0].b.shape)
print(model.network[1].W.shape, model.network[1].b.shape)
print(model.network[2].W.shape, model.network[2].b.shape)
print(model.network[3].W.shape, model.network[3].b.shape)

Out -->
(6, 30) (1, 6)
(8, 30) (1, 8)
(10, 30) (1, 10)
(2, 30) (1, 2)
```
### Propagation vers l'avant (Forward Propagation)
Une fonction qui effectue un passage complet vers l'avant à travers le réseau.
```python
 def _forwardprop(self, data):
        """
        Performs one full forward pass through network
        """
        A_curr = data  # current activation result

        # iterate over layers Weight and bias
        for layer in self.network:
            # calculate forward propagation for specific layer
            # save the ouput in A_curr and transfer it to the next layer
            A_curr = layer.forward(A_curr)
        return A_curr
```
Nous transmettons la sortie de la couche précédente comme entrée à la suivante, désignée par **A_prev**.

Nous stockons les entrées et la somme pondérée dans la mémoire du modèle. Ceci est nécessaire pour effectuer la passe en arrière.

## Couches (Layers) - Forward Pass
### Fonctions d'Activation
```python
def relu(self, inputs):
    """
    ReLU Activation Function
    """
    return np.maximum(0, inputs)

def softmax(self, inputs):
    """
    Softmax Activation Function
    """
    exp_scores = np.exp(inputs)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs
```
**N'oubliez pas qu'il s'agit de fonctions par élément.**

*ReLu*

Utilisé dans les couches cachées. La fonction et le graphique ont été mentionnés dans la section aperçu. Voici ce qui se passe lorsque nous appelons np.maximum().
```
`if input > 0:
 return input
else:
 return 0
```

*Softmax*

Utilisée dans la couche finale. Cette fonction prend un vecteur d'entrée de k valeurs réelles et le convertit en un vecteur de k probabilités dont la somme est égale à un.
\overrightarrow
### $$\sigma (\overrightarrow{Z})_i = \frac{ e^{Z_i} }{\mathcal{D}}$$
### $$\mathcal{D} = \sum_{j=1}^K e^{z_j}$$

où:

### $$e^{z_i} = np.exp(z) au \~ ieme \~ element$$

### $$\sum_j^k (e^{Z_j}) = somme \~ de \~ tous \~ les \~ e^{z_j}$$

### Forward Propagation sur une couche unique:
la fonction *forward()* de la class DenseLayer:
```python
def forward(self, inputs):
        """
        Single Layer Forward Propagation
        """
        Z_curr = np.dot(inputs, self.W.T) + self.b
        A_curr = self.activation(inputs=Z_curr)
        return A_curr
```
Avec:
* inputs = A_prev
* W = matrice de poids de la couche actuelle
* b = vecteur de biais de la couche actuelle
* activation = fonction d'activation de la couche courante

Nous appelons cette fonction dans la méthode **_forwardprop** du réseau et passons les paramètres du réseau en entrée.

###Test de Forward Pass
```
out = model._forwardprop(Xs)
print('SHAPE:', out.shape)
print('Probabilties at idx 0:', out[0])
print('SUM:', sum(out[0]))

Out -->
SHAPE: (569, 2)
Probabilties at idx 0: [0.03732996 0.96267004]
SUM: 1.0
```
Parfait. Tout se met en place ! Nous avons 569 instances mappées à nos 2 classes, et une distribution de probabilité pour chaque instance dont la somme est égale à 1.

## Réseau (Network)
### Backpropagation
Une fonction qui effectue un passage complet en arrière dans le réseau.
```python
def _backprop(self, predicted, actual):
        """
        Performs one full backward pass through network
        """
        num_samples = len(actual)

        # calculate loss derivative of our algorithm
        dscores = predicted
        dscores[range(num_samples), actual] -= 1
        dscores /= num_samples

        dA_curr = dscores
        for layer in reversed(self.network):
            # calculate backward propagation for specific layer
            dA_curr = layer.backward(dA_curr)
```
Remarquez que nous commençons par la couche de sortie et passons à la couche d'entrée.

## Couches - Backward Pass
### Dérivée des fonctions d'Activation

### $$\partial Z^{[L]} = \partial A^{[L]} * g'(Z^{[L]})$$

### Backpropagation pour une couche

dans la Class DenseLayer:
```python
def backward(self, dA_curr):
        """
        Single Layer Backward Propagation
        """
        dZ = self.activation_der(dA_curr, self.mZ)
        dW = np.dot(self.mI.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = np.dot(dZ, self.W)

        self.dW, self.db = dW, db #+ self.reg.regularize(W_curr)
        return dA
```
Cette fonction rétropropage les gradients à chaque paramètre dans une couche.
```python
def relu_prime(dA, Z):
    """
    ReLU Derivative Function
    """
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ
```

### Entraînement du model
```python
def _get_accuracy(self, predicted, actual):
        """
        Calculate accuracy after each iteration
        """
        # for each sample get the index of the maximum value and compare it to the actual set
        # then compute the mean of this False/True array
        return float(np.mean(np.argmax(predicted, axis=1)==actual))
    
    def _calculate_loss(self, predicted: np.ndarray, actual: np.ndarray):
        """
        Calculate cross-entropy loss after each iteration
        """
        samples = len(actual)
        temp = predicted[range(samples, actual.astype(int))]
        temp[temp <= 0] = 1e-15
        correct_logprobs = -np.log(temp)
        data_loss = np.sum(correct_logprobs) / samples
        return float(data_loss)
```
Il est maintenant temps d'effectuer une mise à jour des paramètres après chaque itération (epoch). Implémentons la méthode **_update.**

Dans le réseau (**Network**) chaque couche (**layer**) se met à jour:
```python
def _update(self, lr=0.01):
        """
        Update the model parameters --> lr * gradient
        """
        for layer in self.network:
            # update layer Weights and bias
            layer.update(lr)
```
dans les couches :
```python

def update(self, lr: float):
        if self.regul == 'l2':
            _lambda = 0.5
            m = len(self.dw)
            self.dW += (_lambda / (2 * m)) * np.sum(self.dW ** 2)

        self.W -= lr * self.dW.T
        self.b -= lr * self.db
 ```
 A noter que j'ai rajouté la possibilité de faire une régularisation l2 (faudra peut-etre faire des class TODO)
 
### Test
```
model = Network()
model.add(DenseLayer(6, 'relu'))
model.add(DenseLayer(8, 'relu'))
model.add(DenseLayer(10, 'relu'))
model.add(DenseLayer(2, 'softmax'))

model._compile(Xs)

model.train(Xs, y, 200)

Out -->
EPOCH: 0, ACCURACY: 0.37258347978910367, LOSS: 1.445556163784821
EPOCH: 20, ACCURACY: 0.3936731107205624, LOSS: 0.7303298575724991
EPOCH: 40, ACCURACY: 0.664323374340949, LOSS: 0.6109264451431674
EPOCH: 60, ACCURACY: 0.875219683655536, LOSS: 0.5603079180980095
EPOCH: 80, ACCURACY: 0.8910369068541301, LOSS: 0.5185322121582634
EPOCH: 100, ACCURACY: 0.8963093145869947, LOSS: 0.46985917826143403
EPOCH: 120, ACCURACY: 0.8927943760984183, LOSS: 0.4184126056024643
EPOCH: 140, ACCURACY: 0.9068541300527241, LOSS: 0.3757569072836975
EPOCH: 160, ACCURACY: 0.9226713532513181, LOSS: 0.3421094901271346
EPOCH: 180, ACCURACY: 0.929701230228471, LOSS: 0.31399237195126795
```
### Résultats:
![Figure_1](https://user-images.githubusercontent.com/25301163/214864741-5de050cc-7543-474e-9aeb-bbfdd564bcf3.png)
Bien sur, maintenant reste à chercher la bonne architecture et travailler sur les param

Articles interressant sur les fonction d'activations [ici](https://inside-machinelearning.com/fonction-dactivation-comment-ca-marche-une-explication-simple/)
