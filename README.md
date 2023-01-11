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
$$X \to \fbox{Couche} \to Y$$

### Propagation vers l'avant (Forward propagation)
Nous pouvons déjà souligner un point important, à savoir que la sortie d'une couche est l'entrée de la couche suivante.

$$X \to \fbox{Couche\~1} \to \fbox{Couche\~2} \to \fbox{Couche\~3} \to Y\~E$$

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

$$\frac{\partial E}{\partial X} \leftarrow \fbox{Couche} \leftarrow \frac{\partial E}{\partial Y}$$
Rappelez-vous que E est un **scalaire** (un nombre) et que X et Y sont des **matrices**.
