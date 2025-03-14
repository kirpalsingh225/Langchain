from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

model2 = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')


prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text}',
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 short question answers from the following text \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables = ["notes", "quiz"]
)


parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes" : prompt1 | model1 | parser,
        "quiz" : prompt2 | model2 | parser
    }
)

merged_chains = prompt3 | model1 | parser

chain = parallel_chain | merged_chains


text = """class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)[source]
Ordinary least squares Linear Regression.

LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

Parameters:
fit_interceptbool, default=True
Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

copy_Xbool, default=True
If True, X will be copied; else, it may be overwritten.

n_jobsint, default=None
The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

positivebool, default=False
When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.

Added in version 0.24.

Attributes:
coef_array of shape (n_features, ) or (n_targets, n_features)
Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.

rank_int
Rank of matrix X. Only available when X is dense.

singular_array of shape (min(X, y),)
Singular values of X. Only available when X is dense.

intercept_float or array of shape (n_targets,)
Independent term in the linear model. Set to 0.0 if fit_intercept = False.

n_features_in_int
Number of features seen during fit.

Added in version 0.24.

feature_names_in_ndarray of shape (n_features_in_,)
Names of features seen during fit. Defined only when X has feature names that are all strings.

Added in version 1.0.

See also

Ridge
Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients with l2 regularization.

Lasso
The Lasso is a linear model that estimates sparse coefficients with l1 regularization.

ElasticNet
Elastic-Net is a linear regression model trained with both l1 and l2 -norm regularization of the coefficients.

Notes

From the implementation point of view, this is just plain Ordinary Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares (scipy.optimize.nnls) wrapped as a predictor object."""


result = chain.invoke({"text":text})
print(result)