# Description of the task

## Rapport scientifique

### Contexte du projet et description

Avec la croissance démographique mondiale et les défis liés au changement climatique, il est devenu crucial de gérer efficacement les ressources agricoles pour garantir la sécurité alimentaire et préserver l'environnement. La télédétection par satellite offre une solution moderne et efficace pour la surveillance des cultures, en permettant une analyse régulière et étendue des terres agricoles sans nécessiter de visites sur le terrain. En observant la dynamique des cultures à partir d'images satellites, les décideurs peuvent obtenir des informations clés sur les types de cultures plantées, leur état de santé, ainsi que sur les pratiques agricoles adoptées par les agriculteurs, afin de suivre l'évolution de l'utilisation des terres au fil du temps. Cela revêt une importance capitale pour optimiser l'allocation des ressources, améliorer la gestion des cultures, et garantir le respect des politiques agricoles telles que la politique agricole commune (PAC).

L'objectif principal du projet est de développer un système automatique capable de segmenter et de classifier des parcelles agricoles en fonction des types de cultures présentes, en utilisant des séries temporelles d'images satellites. Contrairement à une simple analyse d'image statique, les séries temporelles permettent de suivre les variations saisonnières et phénologiques (changements au cours du cycle de vie des cultures) sur une année entière, ce qui aide à distinguer les cultures entre elles et à détecter les cultures intermédiaires ou les jachères. En utilisant ces données, le modèle peut générer des prédictions précises sur la nature des cultures présentes, permettant ainsi une prise de décision plus éclairée et de meilleurs résultats commerciaux et environnementaux.

La précision des prédictions issues de ce projet peut avoir des applications multiples :

- **Optimisation de la gestion agricole :** Aider les agriculteurs à prévoir les besoins en irrigation, en engrais et en pesticides, en fonction du type de culture et de son état de santé.

- **Suivi des politiques agricoles :** Permettre aux institutions gouvernementales de vérifier le respect des réglementations agricoles, notamment celles liées aux subventions et aux aides de la PAC.

- **Prédiction et analyse des rendements :** Fournir des estimations précoces des rendements pour planifier la chaîne d'approvisionnement et anticiper les prix des denrées alimentaires sur le marché.

- **Environnement et durabilité :** Faciliter la surveillance de l'utilisation des terres et de la biodiversité, et repérer les pratiques agricoles non durables telles que la monoculture excessive.

- **Étude du changement climatique :** suivre de manière rétroactive l’évolution des pratiques agricoles dans un contexte de réchauffement climatique afin de mieux comprendre et agir sur ces changements.

Pour ce faire, le projet utilise des données satellites multi-spectrales (captant la lumière dans 10 longueurs d'ondes différentes), qui permettent de différencier les cultures grâce à leur signature spectrale et leur périodicité annuelle unique. Le modèle développé, un Vision Transformer adapté aux tâches de segmentation et d'analyse temporelle, exploite ces informations pour fournir des prédictions précises et robustes.

### Scope du projet

Le projet se concentre sur la conception et la mise en œuvre d'un modèle capable de segmenter des images satellites basées sur des séries temporelles et de classifier les parcelles agricoles par type de culture. Pour y parvenir, plusieurs étapes ont été définies :

#### 1. Préparation des données :
Les données d'entrée consistent en des images satellites couvrant des zones de 10x10 km, capturées tout au long de l'année. Chaque image comprend 10 bandes spectrales (par exemple, rouge, vert, bleu, proche infrarouge, etc.) qui fournissent des informations riches sur la végétation et le sol.

Les données en sortie sont une segmentation de chaque image avec 20 classes possibles de types de cultures, chaque classe correspondant à un type de culture spécifique.


#### 2. Développement du modèle :
Le choix d'un Vision Transformer (ViT) a été motivé par sa capacité à traiter des données  tout en tenant compte des relations spatiales et temporelles complexes, notamment dans le cas de séries d’images. Contrairement aux réseaux convolutifs traditionnels, le ViT utilise des mécanismes d'attention pour capturer les dépendances à long terme dans les données, ce qui le rend particulièrement adapté aux séries temporelles.
L'architecture du modèle a été adaptée pour intégrer des séquences temporelles d'images en entrée. Cela permet au modèle de comprendre l'évolution des cultures au fil du temps, plutôt que de se baser uniquement sur un instantané statique.


#### 3. Entraînement et validation :
Pour évaluer la performance du modèle de segmentation, nous avons utilisé des métriques couramment employées en analyse d'images : la mean IoU (Intersection over Union).

Cette métrique mesure la similitude entre la prédiction du modèle et la vérité terrain. Pour chaque classe, l'IoU est calculée en divisant l'aire de l'intersection (les pixels correctement prédits comme appartenant à cette classe) par l'aire de l'union (tous les pixels prédits comme appartenant à la classe, plus ceux qui devraient l'être mais qui ne le sont pas). Formellement, cela se traduit par :

IoU = Aire de l'intersection / Aire de l'union

Plus la valeur de l'IoU est élevée, plus la prédiction est proche de la réalité. La **mean IoU** est la moyenne des IoU calculées pour toutes les classes de culture, offrant une mesure globale de la performance du modèle.

Une difficulté majeure dans la classification des cultures réside dans la gestion des classes de cultures sous-représentées. Si certaines classes sont très fréquentes dans les données, d'autres sont beaucoup plus rares. Cela peut entraîner un déséquilibre : le modèle pourrait ignorer ces classes peu représentées ou avoir du mal à les classifier correctement, ce qui serait problématique si des parcelles rares n'étaient jamais bien identifiées.

Pour remédier à ce problème, nous pouvons appliquer une **pondération des classes** lors de l'évaluation du modèle. Cette technique consiste à augmenter l'importance des erreurs commises sur les classes rares. En d'autres termes, si le modèle se trompe sur une parcelle de culture rare, cette erreur est plus "coûteuse" que si la même erreur avait été commise sur une classe fréquente. Cela incite le modèle à accorder plus d'attention aux cultures moins représentées et à apprendre à mieux les reconnaître, assurant ainsi une classification plus précise et plus équitable de toutes les parcelles agricoles, même les plus rares.

Ce processus devrait permettre d'obtenir un modèle plus robuste et capable de généraliser sur l'ensemble des types de cultures présents dans les données, garantissant que toutes les catégories de parcelles soient bien prises en compte, quel que soit leur niveau de représentativité.

### Présentation du groupe
Le projet a été réalisé par un groupe de quatre étudiants en dernière année à l’École des Mines de Paris, dans le cadre d’un data challenge organisé par Capgemini. Le challenge nous a offert l’opportunité d’aborder des problèmes concrets en utilisant des techniques avancées de data science et d’intelligence artificielle. Le groupe a combiné ses compétences en Deep Learning, traitement d'images, et analyse de séries temporelles pour relever ce défi ambitieux.

### Gestion des tâches
Le projet s'est déroulé sur une période d’une semaine intensive. Pour assurer une coordination efficace, l'équipe a travaillé majoritairement en présentiel durant la journée, permettant de discuter des choix techniques, de résoudre les problèmes rencontrés, et de tester rapidement de nouvelles idées. Le soir, chaque membre poursuivait le travail de son côté, afin de progresser sur des aspects spécifiques du projet.

Afin de faciliter la collaboration et le partage de code, nous avons utilisé un <a href="https://github.com/AlexiaALLAL/hackathon_capgemini"> repository GitHub </a> pour centraliser le développement. Chaque membre pouvait ainsi intégrer ses contributions, permettant un suivi continu et une mise à jour régulière du projet. Cette méthode de travail en équipe a favorisé une grande réactivité et a permis de produire un prototype fonctionnel en un temps très court.

Nous avons aussi essayé de paralléliser au maximum les recherches, et quand c’était impossible, de lancer des recherches dans plusieurs directions différentes en même temps, quitte à abandonner rapidement celles qui n’auraient pas porté de fruits. Cette méthode nous à permis d’explorer plusieurs architectures en parallèle et de conclure sur les avantages et les inconvénients de chacune d’entre elles.

 
## Gestion de projet

### Compréhension des données
La première étape du projet a consisté à comprendre en profondeur la structure et la nature des données disponibles. Les données fournies comprenaient des **séries temporelles d'images satellites**, représentant des zones de 10x10 km tout au long d'une année. Chaque image contient des informations sur **10 bandes spectrales différentes**, telles que le rouge, le vert, le bleu, et le proche infrarouge. Ces bandes spectrales permettent d'observer divers aspects des cultures, comme la santé végétale ou l'humidité du sol, à différentes périodes de l'année.

Les données d'entrée peuvent être organisées sous la forme d’un **tenseur de taille (N, T, C, H, W)** :

- **N** représente le nombre de différentes zones géographiques observées (nombre de parcelles),

- **T** est le nombre de fois où chaque zone a été photographiée au cours de l'année. Pour garantir une cohérence des données, toutes les séries temporelles ont été complétées par des **images nulles** lorsque nécessaire, de manière à obtenir exactement **T=61** images pour chaque zone,

- **C** est le nombre de canaux spectraux (bandes spectrales), avec **C=10** pour les 10 longueurs d'onde capturées par les satellites,

- **H** et **W** sont respectivement la hauteur et la largeur des images, fixées à **128x128 pixels**.

Le modèle de segmentation que nous avons développé prend ce tenseur comme entrée et produit en sortie un **tenseur de taille (B, H, W)**, où **B** représente le batch size utilisé pendant l’entraînement ou la prédiction, et **H** et **W** correspondent toujours à la taille des images de 128x128 pixels. Chaque pixel de la sortie est associé à l'une des **20 classes** de types de culture, permettant ainsi de générer des cartes de segmentation des cultures sur chaque parcelle analysée.

Cette structure de données claire et standardisée a facilité la manipulation et l'analyse des séries temporelles complexes


### Prétraitement des données
Le prétraitement des données est une étape essentielle dans tout projet de modélisation, mais dans notre cas, cette phase a été simplifiée grâce à la qualité des données fournies. Les données étaient déjà **nettoyées et prétraitées** par Capgemini, ce qui a grandement facilité notre travail. Nous n’avons pas eu besoin de procéder à des étapes supplémentaires de filtrage ou de normalisation, et avons ainsi pu insérer directement les données dans notre modèle pour l'entraînement et la validation.

Cette situation nous a permis de nous concentrer davantage sur les aspects de modélisation et d’optimisation du modèle, plutôt que sur le traitement des données brutes.

### Modélisation
La modélisation a constitué le cœur du projet. Pour cette tâche, nous avons opté pour un **Vision Transformer (ViT)**, adapté pour traiter à la fois des images et des séries temporelles complexes. 

Voici les principales étapes de cette phase :

- **Choix du modèle :** Le ViT a été choisi pour sa capacité à exploiter des **mécanismes d'attention** permettant de capturer les relations spatiales et temporelles entre les différents pixels d’une image. Contrairement aux réseaux de neurones convolutifs traditionnels, le ViT offre une plus grande flexibilité pour apprendre des dépendances à long terme, ce qui est essentiel pour notre tâche de segmentation des cultures.

- **Difficultés et adaptations nécessaires :** La principale difficulté rencontrée durant cette phase a été d’adapter le ViT pour répondre aux spécificités de notre projet :
    - **Adaptation à la segmentation d’images :** Par défaut, le ViT est conçu pour classifier des images complètes en une seule catégorie. Nous avons dû modifier sa structure pour qu'il puisse produire une classe par pixel de l’image d’entrée, permettant ainsi de générer des cartes de segmentation précises des parcelles agricoles.
    
    - **Prise en compte des 10 canaux :** Contrairement aux images standard en RGB (3 canaux), nos données contiennent 10 bandes spectrales. Il a fallu adapter le modèle pour qu'il puisse intégrer et traiter ces 10 canaux simultanément, en s'assurant que chaque bande spectrale contribue efficacement à la segmentation.
    
    - **Adaptation aux séries temporelles :** Le modèle devait également comprendre la dynamique temporelle des images, c’est-à-dire la façon dont les cultures évoluent au fil du temps. Pendant nos recherches, nous avons tenté de modifier le ViT pour qu'il prenne en entrée des séquences temporelles d'images, permettant de capturer ces variations et d'améliorer la précision de la prédiction finale. La technique choisie a été de concaténer les embeddings des images de chaque série temporelle pour former un embedding de la série. Malheureusement, le modèle ainsi développé prend des dimensions trop grandes pour que notre puissance de calcul suffise à l'entraîner

- **Entraînement et évaluation :** Pour entraîner le modèle, nous avons utilisé le jeu de données étiqueté fourni par Capgemini comprenant différentes séries temporelles d'images satellites. Les performances ont été mesurées à l’aide de métriques telles que la mean IoU, en tenant compte des pondérations pour les classes rares, afin de s'assurer que le modèle pouvait détecter même les types de cultures les moins représentés.


### Stratégie de déploiement
L’entraînement du modèle nécessitant une puissance de calcul importante, nous avons utilisé des GPU pour accélérer le processus. En raison des limitations de nos ordinateurs personnels, nous avons opté pour Google Colab et Kaggle, deux plateformes offrant un accès gratuit à des GPUs. Cette solution nous a permis d’entraîner nos modèles efficacement et de réaliser plusieurs itérations pour améliorer les performances du modèle.

Pour la suite du déploiement, nous envisageons de designer une interface utilisateur permettant à un utilisateur peu familier avec les environnement de code d’obtenir des résultats. Comme le modèle est destiné à être utilisé par un nombre relativement restreint d’utilisateurs, une simple interface devrait suffire à son utilisation.


## Conclusion

Lors de ce projet, notre équipe a été amenée à explorer différentes technologies à même de produire des résultats dans un temps limité et avec une puissance de calcul restreinte. Le travail de groupe à été source d’innovation et à permis d’explorer plusieurs architectures pour le modèle de manière simultanée. Finalement, l’architecture choisie est un ViT de classification, adapté à la tâche de segmentation de notre jeu de données. Si les performances de notre modèle (environ 8% d’IoU sur le jeu de test) ne sont pas suffisantes pour qu’il serve son but initial, elles ont probablement été bridées par le manque de temps de développement et d'entraînement auquel nous avons dû faire face. 

Ainsi, les axes de recherches toujours actifs sont de parvenir à implémenter la dépendance spatiale dans notre modèle de manière optimisée (le blocage aujourd’hui réside dans la puissance de calcul nécessaire à entraîner notre implémentation), ainsi que de parvenir à implémenter un ViT pré entraîné sur une tâche de segmentation sur des images satellites, afin de capitaliser sur les recherches passées.

