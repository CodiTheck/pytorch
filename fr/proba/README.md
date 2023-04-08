## Distribution de probabilité
![](https://img.shields.io/badge/lastest-2023--04--08-success)
![](https://img.shields.io/badge/status-en%20r%C3%A9daction%20-yellow)

Les probabilités et les variables aléatoires font partie intégrante du calcul
dans un framework de calcul de graphes comme PyTorch. Il est essentiel
de comprendre les probabilités et les concepts associés. Dans ce chapitre,
on va parler des distributions de probabilités et de leur mise en œuvre avec
PyTorch, ainsi que l'interprétation des résultats des tests.

En probabilités et en statistiques, une variable aléatoire est également 
connue sous le nom de **variable stochastique**, dont le résultat dépend
d'un phénomène purement stochastique. Il existe différents types de
distributions de probabilités, dont la distribution normale, la distribution
binomiale, la distribution multinomiale et la distribution de Bernoulli.
Chaque distribution statistique a ses propres propriétés.

Le module `torch.distributions` contient des distributions de probabilité et
des fonctions d'échantillonnage. Chaque type de distribution a sa propre
importance dans un graphe de calcul. Le module "distributions" contient
les distributions suivantes :

- binomiale,
- Bernoulli,
- bêta,
- catégorielle,
- exponentielle,
- normale et Poisson.

<details id="table-content" open>
    <summary>Table des Contenus</summary>
    <ul>
        <li><a href="#pourquoi-pytorch">Pourquoi PyTorch ?</a></li>
        <li><a href="#création-de-tenseur">Création de Tenseur</a>
            <ul>
                <li><a href="#scalaire">Scalaire</a></li>
                <li><a href="#vecteur">Vecteur</a></li>
            </ul>
        </li>
    </ul>
</details>


<!--### Tenseurs d'échantillonnage
L'initialisation des poids est une tâche importante dans la formation
d'un réseau neuronal et de tout type de modèle d'apprentissage profond,
tel qu'un réseau neuronal convolutionnel (CNN) un réseau neuronal profond (DNN)
et un réseau neuronal récurrent (RNN). La question se pose toujours de savoir
comment initialiser les poids.

L'initialisation des poids peut être effectuée en utilisant différentes
méthodes, notamment l'initialisation aléatoire des poids. Pour exécuter
un réseau neuronal, un ensemble de poids initiaux doit être transmis
au différentes couches afin de calculer la fonction de perte (et,
donc le score ou pourcentage de prédiction peut être calculé). Le choix
d'une méthode d'initialisation dépend du type de données, de la tâche et
de l'optimisation requise pour le modèle. On va donc examiner tous les types
d'approches pour initialiser les poids.-->



<br/>
<br/>

<!-- - Je passe à la session **suivante** : -->
<!-- [Distribution de probabilité](./proba/README.md) -->
[<--](../tensor/README.md) Je reviens à la session **précédente** :
[Introduction au Tenseur](../tensor/README.md)
