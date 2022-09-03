
class PopulationBasedTraining(object):
    """
    """


    def __init__(self, parameters_range, initial_population_size = 100, model = None):
        """
        """
        
        self.parameters_range = dict(parameters_range).copy(),
        self.parameters_range = self.parameters_range[0]
        self.population_size = initial_population_size
        self.model = model

    def spawn(self, ):
        """
        """
        
        population_parameters = []
        population = []
        
        for i in range(self.population_size):
            population_parameters.append({k: random.choice(v) for k,v in self.parameters_range.items()})

        for params in population_parameters:
            model_ = self.model
            population.append(model_(**params))

        return population, population_parameters

    @staticmethod
    def fit_single(model, X, y, sample_weight=None):
        """
        fit single model with provided X and y data
        """

        model.fit(X, y, sample_weight)

        return model


    def fit_many(self, models, X, y, sample_weight=None):
        """
        """

        fit_single_part = partial(self.fit_single, X=X, y=y, sample_weight=sample_weight)

        models_fitted = list(map(fit_single_part, models))

        return models_fitted


    @staticmethod
    def score_single(model, X, y, scoring_fcn, sample_weight = None):
        """
        score single model with provided model, X and y data
        scoring_fcn - sklearn.metrics.roc_auc_score
        """

        y_hat = model.predict(X)
        if sample_weight:
            score = scoring_fcn(
                y, 
                y_hat, 
                average = 'weighted',
                sample_weight=sample_weight
                )
        else:
            score = scoring_fcn(y, y_hat)

        return score


    def score_many(self, models, X, y, scoring_fcn, sample_weight = None):
        """
        
        """

        score_single_part = partial(self.score_single, X=X, y=y, scoring_fcn=scoring_fcn, sample_weight=sample_weight)

        scores = list(map(score_single_part, models))

        return scores


    @staticmethod
    def fitness(scores, top_k = 100):
        """
        """

        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]


    def cross(self, parent1, parent2, dominance=0.5):
        """
        """

        dominance_size = int(np.ceil(dominance*len(self.parameters_range.keys())))
        # print(parent1)
        params_parent1 = {k: parent1[k] for k in list(parent1)[:dominance_size]}
        # print(params_parent1)
        # print('here')
        params_parent2 = {k: parent2[k] for k in list(parent2)[dominance_size:]}
        # print(params_parent2)
        # print('here 2')
        
        child = params_parent1.copy()
        child.update(params_parent2)
        # print('child')
        # print(child)

        return child

    def mutate_single(self, model_params_dict):
        """
        """

        list_params = list(model_params_dict.keys())
        params_to_mutate = random.choices(list_params, k = random.randint(1, len(list_params)))
        
        model_params_dict = model_params_dict.copy()
        for param in params_to_mutate:
            # print('--')
            # print(param)
            # print('--')
            # print(self.parameters_range)
            # print('--')
            # print(self.parameters_range[param])
            # print('--')
            # print(self.parameters_range)
            # print('--')
            # print(random.choice(self.parameters_range[param]))
            # print('--')
            # print(model_params_dict)
            # print('--')
            model_params_dict[param] = random.choice(self.parameters_range[param])
        
        return model_params_dict


    def mutate_many(self, population):
        """
        """

        return list(map(self.mutate_single, population))


    def run(self, X, y, X_valid, y_valid, scoring_fcn, top_k, dominance, max_epochs, mutate_condition, max_score_th=0.95, sample_weight=None):
        """
        """

        epoch = 1
        best_scores = []
        best_models = {}

        # 1st spawn intial population and set epoch to 1
        population, population_parameters = self.spawn()

        # fit initial population
        fitted_models = self.fit_many(population, X, y, sample_weight=sample_weight)

        # score initial population
        population_scores = self.score_many(fitted_models, X_valid, y_valid, scoring_fcn, sample_weight=sample_weight)

        # best score from prev population
        best_model_idx = self.fitness(population_scores, top_k = 1)
        best_scores.append([population_scores[x] for x in best_model_idx])
        
        print(epoch)
        print('best score: ', best_scores[-1])
        
        best_models[epoch] = {
            'model': fitted_models[best_model_idx[-1]],
            'parameters': population_parameters[best_model_idx[-1]]
        }

        # fitness
        fitness_idx = self.fitness(population_scores, top_k = top_k)


        # select children and cross for children
        parents_models = [fitted_models[x] for x in fitness_idx]
        parents_params = [population_parameters[x] for x in fitness_idx]

        children = []

        for p1, p2 in list(itertools.product(parents_params, parents_params)):
            # print(p1)
            # print(p2)
            children.append(self.cross(p1, p2, dominance))
            children.append(self.cross(p2, p1, 1.0-dominance))
        
        # print(children)
        
        population = []
        for child in children:
            population.append(self.model(**child))
        
        population_parameters = children.copy()
        fitted_models = self.fit_many(population, X, y, sample_weight=sample_weight)
        population_scores = self.score_many(fitted_models, X_valid, y_valid, scoring_fcn, sample_weight=sample_weight)
        
        # best score from prev population
        best_model_idx = self.fitness(population_scores, top_k = 1)
        best_scores.append([population_scores[x] for x in best_model_idx])

        epoch = 2
        print(epoch)
        print('best score: ', best_scores[-1])

        best_models[epoch] = {
            'model': fitted_models[best_model_idx[-1]],
            'parameters': population_parameters[best_model_idx[-1]]
        }

        # fitness
        fitness_idx = self.fitness(population_scores, top_k = top_k)

        not_stop = True

        while(not_stop):

            epoch += 1
            print(epoch)

            # select children and cross for children
            parents_models = [fitted_models[x] for x in fitness_idx]
            parents_params = [population_parameters[x] for x in fitness_idx]

            children = []

            for p1, p2 in list(itertools.product(parents_params, parents_params)):
                children.append(self.cross(p1, p2, dominance))
                children.append(self.cross(p2, p1, 1.0-dominance))
            
            # print(children)
            # mutate children's population
            best_score_diff = best_scores[-1][-1] - best_scores[-2][-1]
            if best_score_diff < mutate_condition:
                children = self.mutate_many(children)
                
            population_parameters = []
            for child in children:
                population.append(self.model(**child))
                
            population_parameters = children.copy()
            fitted_models = self.fit_many(population, X, y, sample_weight=sample_weight)
            population_scores = self.score_many(fitted_models, X_valid, y_valid, scoring_fcn, sample_weight=sample_weight)
            
            # best score from prev population
            best_model_idx = self.fitness(population_scores, top_k = 1)
            best_scores.append([population_scores[x] for x in best_model_idx])
            
            best_scores_ = [x[-1] for x in best_scores]
            best_score = max(best_scores_)

            print('best score: ', best_scores[-1])

            best_models[epoch] = {
                'model': fitted_models[best_model_idx[-1]],
                'parameters': population_parameters[best_model_idx[-1]]
            }

            if epoch == max_epochs:
                not_stop = False
                return best_models

            if best_score >= max_score_th:
                non_stop = False
                return best_models

            