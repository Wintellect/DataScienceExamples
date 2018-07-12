using Microsoft.ML.Runtime.Api;

namespace MLNetShared
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}
