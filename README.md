# 5MLDE_RENDU

## 1 - Build les images avec docker : 

Lancer le fichier "build_images.ps1"

ou 

Ouvrir powershell dans le dossier et écrire les commandes suivantes : 
> docker build -t hearth_attack_jupyter ./infra/jupyter

> docker build -t hearth_attack_mlflow ./infra/mlflow_server

> docker build -t hearth_attack_api ./infra/api


## 2 - Créer un réseau :

Exécuter la commande suivante :
>docker network create --driver bridge hearth_attack_netw


## 3 - Executer les containers pour jupyter et mlflow :

Lancer le fichier "run_containers.ps1"

ou 

Ouvrez un powershell et écrire les commandes suivantes :
>docker run -it -d --rm -p 5001:5000 -v ${PWD}/infra/mlflow_server/local:/mlflow --network hearth_attack_netw --name mlflow hearth_attack_mlflow

[//]: # (>docker run -it -d -p 8002:8002 -v ${PWD}/infra/mlflow_server/local:/mlflow --network hearth_attack_netw --name api hearth_attack_api)

>docker run -it --rm --user root -p 10000:8888 -p 8000:8000 -p 4200:4200 -v ${PWD}/infra/mlflow_server/local:/mlflow -e JUPYTER_ENABLE_LAB=yes -e JUPYTER_TOKEN=docker -e MLFLOW_TRACKING_URI=http://mlflow:5000 --network hearth_attack_netw --name jupyter -d hearth_attack_jupyter

## 4 - Déployer le modèle :

Aller sur l'url http://localhost:10000.

Se connecter au jupyter grâce au mot de passe "MLOPS"

Lancer un nouveau terminal dans l'environnement jupyter et taper la commande suivante :
>python orchestration.py


## 5 - Créer le container pour l'api :

Ouvrir un powershell et écrire les commandes suivantes :
>docker run -it -d -p 8002:8002 -v ${PWD}/infra/mlflow_server/local:/mlflow --network hearth_attack_netw --name api hearth_attack_api


## 6 - Faire des requêtes à l'api :

Il existe deux requêtes possibles : 
- Une requête http://localhost:8002/ en GET permettant de voir si l'api fonctionne correctement.
- Une requête http://localhost:8002/predict en POST avec un objet contenant les informations de la personne dans le body afin de prédire si elle a des chances d'avoir une crise cardiaque.

## 7 - Executer l'entrainement du modèle regulierement
Aller sur l'url http://localhost:10000.

Lancer un nouveau terminal dans l'environnement jupyter et taper la commande suivante :

>python schedule.py

Ce script s'executera tous les dimanches et enregistera le modèle en production si 
ses reusltats sont superieurs aux précédents.