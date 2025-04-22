from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Union, List, Optional, Dict, Any
from ml.classification import knn, decision_tree, naive_bayes, neural_network,random_forest,svm
from ml.regression import linear_regression, multiple_regression
from ml.clustering import kmeans, dbscan, agglomerative
from ml.association import apriori

router = APIRouter()

# === Param Models ===
class KNNParams(BaseModel):
    n_neighbors: int = 3

class DecisionTreeParams(BaseModel):
    criterion: str = "gini"
    max_depth: Optional[int] = None

class NaiveBayesParams(BaseModel):
    model_type: Optional[str] = "auto"

class NeuralNetworkParams(BaseModel):
    hidden_layer_sizes: List[int] = [100]
    max_iter: int = 200

class RandomForestParams(BaseModel):
    n_estimators: int = 100
    max_depth: Optional[int] = None

class SVMParams(BaseModel):
    kernel: str = "rbf"
    c: float = 1.0

class DBSCANParams(BaseModel):
    eps: float = 0.5
    min_samples: int = 5

class KMeansParams(BaseModel):
    n_clusters: int = 3

class AgglomerativeParams(BaseModel):
    n_clusters: int = 2

class AprioriParams(BaseModel):
    min_support: float = 0.5
    min_confidence: float = 0.7

# === Request Model ===
class AnalyzeRequest(BaseModel):
    algorithm: str
    data_path: str
    target_column: Optional[str] = None
    parameters: Dict[str, Any]

# === Endpoint ===
@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    algo = request.algorithm.lower()
    result = {}

    try:
        if algo == "knn":
            params = KNNParams(**request.parameters)
            result = knn.run(
                data_path=request.data_path,
                target_column=request.target_column,
                n_neighbors=params.n_neighbors
            )

        elif algo == "decision_tree":
            params = DecisionTreeParams(**request.parameters)
            result = decision_tree.run(
                data_path=request.data_path,
                target_column=request.target_column,
                criterion=params.criterion,
                max_depth=params.max_depth
            )

        elif algo == "naive_bayes":
                params = NaiveBayesParams(**request.parameters)
                result = naive_bayes.run(
                data_path=request.data_path,
                target_column=request.target_column,
                model_type=params.model_type
            )


        elif algo == "neural_network":
            params = NeuralNetworkParams(**request.parameters)
            result = neural_network.run(
                data_path=request.data_path,
                target_column=request.target_column,
                hidden_layer_sizes=params.hidden_layer_sizes,
                max_iter=params.max_iter
            )
        
        
        elif algo == "random_forest":
            params = RandomForestParams(**request.parameters)
            result = random_forest.run(
                data_path=request.data_path,
                target_column=request.target_column,
                n_estimators=params.n_estimators,
                max_depth=params.max_depth
            )

        elif algo == "svm":
            params = SVMParams(**request.parameters)
            result = svm.run(
                data_path=request.data_path,
                target_column=request.target_column,
                kernel=params.kernel,
                c=params.c
            )

        elif algo == "linear_regression":
            result = linear_regression.run(
                data_path=request.data_path,
                target_column=request.target_column
            )

        elif algo == "multiple_regression":
            result = multiple_regression.run(
                data_path=request.data_path,
                target_column=request.target_column
            )

        elif algo == "kmeans":
            params = KMeansParams(**request.parameters)
            result = kmeans.run(
                data_path=request.data_path,
                n_clusters=params.n_clusters
            )

        elif algo == "dbscan":
            params = DBSCANParams(**request.parameters)
            result = dbscan.run(
                data_path=request.data_path,
                eps=params.eps,
                min_samples=params.min_samples
            )

        elif algo == "agglomerative":
            params = AgglomerativeParams(**request.parameters)
            result = agglomerative.run(
                data_path=request.data_path,
                n_clusters=params.n_clusters
            )

        elif algo == "apriori":
            params = AprioriParams(**request.parameters)
            result = apriori.run(
                data_path=request.data_path,
                min_support=params.min_support,
                min_confidence=params.min_confidence
            )

        else:
            raise HTTPException(status_code=400, detail="Algorithme non supporté.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}")

    return result
