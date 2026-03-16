import numpy as np
# todo : initialize configuration randomly for a 10-dimensional problem

# la configuration de base dépend du problème. elle représente l'espace des antécédents
# si on a une fonction de cout continue, on peut utiliser un décimal directement
# Si la fonction est à plusieurs variable, il faut un tableau pour ces variables, une variables = 1 élément
# Si on représente avec qubo, on utilise un tableau de 0 et 1 dont la chaine représente l'espace.
# plusieurs manière de jouer dessus puisqu'il faut de l'aléatoire pour les changements de config. 
# en qubo, on peut générer des décimaux et arrondir, ou donne une proba de flip le bit, ou faire du random.choice([0,1]) pour chaque bit.


# test simple : 
conf_init = np.random.rand(0,12)

# todo : define an objective function to minimize, for example, the sphere function
def objective_function(x):
    return np.sum(x**2 -4)

temperature = 100.0

explored = []
explored.append(objective_function(conf_init))

cooling_rate=0.95

while temperature > 0.01:
    conf = conf_init
    new_conf = conf + np.random.normal(0, 0.1, size=conf.shape)  # detail np.random.normal(0, 0.1, size=conf.shape) normal noise autourde notre nombre
    # calcul de la proba de garder cette nouvelle config, on utilise la formule de Metropolis : p = exp(-delta_E / T) où delta_E  inspirée de la physique statistique
    # l'energie est la cost function
    delta_E = objective_function(new_conf) - objective_function(conf)
    p = np.exp(-delta_E/temperature)
    if(np.random.rand()>=p):
        conf = new_conf
    temperature = temperature*cooling_rate
    explored.append(objective_function(conf))
    s= np.min(explored)+4.0
if(s<=0):
    print(np.sqrt(-s))
else :
    print(np.sqrt(s))    

