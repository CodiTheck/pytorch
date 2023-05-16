## Introduction
![](https://img.shields.io/badge/lastest-2023--05--16-success)
<!-- ![](https://img.shields.io/badge/status-en%20r%C3%A9daction%20-yellow) -->

Ce cours ne remplace pas tout ce qui se trouve sur cette
[page d'accueil](https://pytorch.org/docs/stable/index.html).
Il devrait être ta référence pour tout ce qui concerne Pytorch.
Avant que je ne commence les bonnes choses, je te propose de jetter d'abord
un coup d'œil aux
[papiers with code](https://paperswithcode.com/trends). Au cas où, tu ne sais
pas ce que c'est, il s'agit d'un site web qui recense les frameworks les plus
récents et les plus importants sur le machine learning et qui indique s'ils
contiennent ou non des dépôts de code.

<div align="center">

###### FIGURE 01
![](./images/paper_with_code.png)

</div>
<p align="center">
<i>
<ins>Figure 01</ins> : Voici à quoi il ressemble à l'heure où je rédige ce
chapitre introductif, lorsque je place ma souris sur le sommet qui se trouve
en haut et à droite du grand rectangle.
</i>
</p>

Sur ce site, tu trouveras des informations sur
d'autres langages, d'autres frameworks de deep learning, *PyTorch*,
*TensorFlow*, *Jax*, *MXNet*, *PaddlePaddle*, le *torch* original, etc.
La courbe orange est pour le framework PyTorch.
Ce dernier est une évolution de torch écrite en Python. Si tu regardes les
informations de Mars 2023, **PyTorch se trouve 61% avec 2400 dépôts de
code**. Donc, il s'agit, de loin, du framework de recherche le plus populaire
en matière de machine learning, utilisé pour écrire le code des algorithmes
d'apprentissage automatique de pointe.

> Wow, la vâche !!! C'est géniale ! :nerd_face:

Là, je sent que tu es motivé ! :blush:

<br/>
<details id="table-content" open>
    <summary>Table des Contenus</summary>
    <ul>
        <li><a href="#pourquoi-pytorch-">Pourquoi PyTorch ?</a></li>
        <li><a href="#utilisation-de-pytorch">Utilisation de PyTorch</a></li>
        <li><a href="#avantages-de-pytorch">Avantages de PyTorch</a></li>
    </ul>

</details>

<div align="center">

[:house: **Retour à l'accueil**](../README.md)

</div>

### Pourquoi PyTorch ?
> Quelle question ! :grin: <br/>
Il n'y a plus à trop réfléchir. D'après les statistiques,
PyTorch écrase tous le reste. Sans l'ombre d'un doute, c'est PyTorch il faut
choisir.

Oui, je sais, après avoir vu les statistiques, pour toi la question n'a plus
lieu d'être. Mais, je vais quand y répondre.

PyTorch est utilisé pour les modèles de vision
par ordinateur des voitures autonomes de Tesla.

<div align="center">

###### FIGURE 02
![](./images/image_01.jpeg)

</div>
<p align="center">
<i>
<ins>Figure 02</ins> : Une voiture autonome.
</i>
</p>

OpenAI, qui est l'une des plus grandes entreprises ouvertes de recherche en
intelligence artificielle (ouverte dans le sens où elle publie un grand
nombre de ses méthodologies de recherche) en a récemment fait l'objet d'un
débat. Mais si tu vas sur [openai.com](https://openai.com), disons qu'il s'agit
de l'une des plus grandes entités de recherche en IA au monde, et qu'ils ont
standardisé dans PyTorch. Car, l'article de blog de janvier 2020 indique
qu'OpenAI est désormais standardisé dans PyTorch. Il y a un repo appelé
l'[**incroyable PyTorch**](https://github.com/ritchieng/the-incredible-pytorch)
qui rassemble tout un tas de projets différents
qui sont construits au-dessus de PyTorch. La beauté de Pytorch réside dans
le fait que l'on peut l'utiliser pour construire des systèmes d'intelligence
artificielle pour l'agriculture.

> C'est vrai ? :grinning:

Biensûr que si. Les robots utilisés en agriculture de nos jour, utilisent
PyTorch pour leur modèle d'apprentissage machine. Si tu tape sur Google
"**AI for AG**", parmit les résultats, tu trouveras ce
[lien](https://medium.com/pytorch/ai-for-ag-production-machine-learning-for-agriculture-e8cfdb9849a1).

PyTorch est une bibliothèque d’apprentissage automatique assez récente, mais
elle dispose d’un grand nombre de manuels et de tutoriels où il est possible
de trouver des exemples. Elle dispose également d’une communauté qui se
développe à pas de géant.

Contrairement à d’autres outils d’apprentissage automatique tels que
Tensorflow, PyTorch fonctionne avec des graphes dynamiques plutôt que
statiques. Cela signifie qu’au moment de l’exécution, les fonctions peuvent
être modifiées. Alors que, dans Tensorflow, il faut tout d’abord définir le
graphe de calcul, puis utiliser la session pour calculer les résultats des
tenseurs, ce qui rend le débogage du code plus difficile et la mise en œuvre
plus fastidieuse.

PyTorch est compatible avec les cartes graphiques (GPU). Il utilise
**CUDA** en interne, une API qui relie le CPU au GPU et qui a été développée
par *NVIDIA*.


### Utilisation de PyTorch
Ce framework est pensé pour minimiser le temps de calcul et exploiter au mieux
les capacités du hardware. Il représente les données et les modèles sous forme
de tableaux multidimensionnels, similaires aux tableaux NumPy, appelés
**tenseurs**. à partir de ces derniers, PyTorch effectue, de manière cachée et
efficace 4 étapes pour entraîner un modèle de réseau de neurones, à savoir :

1. **Construire un graphe de calcul** à partir des tenseurs du réseau de
neurones. Ce qui permet d'avoir une structure dynamique dans laquelle
il est possible de modifier le réseau de neurones (nombre de noeuds, connexions
entre eux, ...) au cour du processus d'apprentissage.

2. **Effectuer les prédictions** ("Forward propagation").

3. **Calculer la perte** ou erreur de prédictions.

4. **Calculer les gradients** de l'erreur par rapport à paramètre du réseau de
neurones. Cette opération s'appelle "Backward propagation".

5. **Mise à jour des paramètres** du réseau avec les différentes valeurs de
gradients calculées en ajustant les tenseurs qui le représentent. Cette
dernière étape permet au réseau d'améliorer la précision de ces futures
prédictions.

Récemment, et en partenariat avec AWS (Amazon Web Services), PyTorch
a devoilé 2 nouvelles fonctionnalités. La première, baptisée `TorchServe`,
permet de gérer efficacement le déploiement de réseaux de neurones déjà
entraînés. La seconde, `TorchElastic`, permet d’utiliser Pytorch sur des
clusters **Kubernetes** tout en étant résistant aux pannes.

> **Kubernetes**, devenu un outil indispensable en Data Science, est une
plateforme open source, conçu pour gérer des conteneurs et des clusters depuis
une interface unique et centralisée. Il est possible de déployer des
conteneurs sur tous types d’environnements : Cloud, machines virtuelles,
machines physiques, etc... afin de créer un réseau de machines virtuelles.
Pour plus d'informations sur Kubernetes, suis cette
[🎞 vidéo](https://youtu.be/NChhdOZV4sY).


### Avantages de PyTorch
Bien que PyTorch présente de nombreux avantages, on va juste se concentrerons
ici sur les quelques-uns.

- **PyTorch est coder en python** : La plupart des travaux liés à
l’apprentissage automatique et à l’Intelligence Artificielle sont effectués
en utilisant le langage Python. Et étant donné que Pytorch a été codé en
Python, alors les développeurs Python devraient se sentir plus à l’aise.

- **Facile à apprendre** : Comme le langage Python, PyTorch est considéré
comme relativement plus facile à apprendre par rapport à d’autres frameworks.
La raison principale est due à sa syntaxe simple et intuitive.

- **Communauté forte** : Bien que PyTorch soit un framework relativement
récent, il a développé très rapidement une communauté dédiée de développeurs.
De plus, la documentation de PyTorch est très organisée et permet aux
débutants de mieux se retrouver.

- **Débogage facile** : PyTorch est profondément intégré à Python, du coup,
plusieurs outils de débogage Python peuvent être facilement utilisés
avec.


<br/>
<br/>
<div align="center">

[:arrow_backward: Accueil](../README.md)
| [**Installation et configuration :arrow_forward:**](../install/README.md)

</div>
