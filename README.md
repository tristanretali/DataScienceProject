# Lancement application

## Via Docker

Tout d'abord, veuillez vérifier que Docker est bien installé sur votre machine via la commande : `docker --version`. 
Si ce n'est pas le cas l'installer.  

Ensuite lancer l'application **Docker Desktop**

Placez vous dans le répoertoire du projet et lancez ces deux commandes :  

```
docker compose build --no-cache  
docker compose up
```

Le **build** va permettre de créer le container et le **up** de le lancer.

L'API sera ensuite exposer à cette adresse : *<http://127.0.0.1:8000/>*

Routes disponible:

```
http://127.0.0.1:8000/docs -> affiche toute les routes disponible et les paramètres requis. 
http://127.0.0.1:8000/api/version/predict -> version est un paramètre disponible avec les valeurs v1 et v2. Puis dans le body de votre requête POST, il faut mettre un JSON comme telle:  
{
  "Year": 2020,
  "Engine Size": 1.6,
  "Fuel Type": "Electric",
  "Transmission": "Manual",
  "Mileage": 45000,
  "Condition": "Used",
 "Model": "Model X"
}
```

## Via FastAPI

Placez vous dans le dossier du projet et saisissez cette commande: `pip install -r requirements-api.txt`

Cela installe les package python nécessaire pour faire fonctionné l'API.  
Ensuite placez vous dans le dossier **api** avec la commande `cd api`. Il ne vous reste plus qu'à lancer l'API avec `uvicorn main:app`

Vous ne devriez pas avoir d'erreur. Pour le fonctionnement des routes c'est le même que énoncé plus haut.

