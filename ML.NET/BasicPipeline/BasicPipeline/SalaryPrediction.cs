using Microsoft.ML.Runtime.Api;

namespace BasicPipeline2
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}
