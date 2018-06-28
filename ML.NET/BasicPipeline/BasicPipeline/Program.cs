using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace BasicPipeline2
{
    class Program
    {
        private const int PREDICTION_YEARS = 8;

        static void Main(string[] args)
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader("SalaryData.csv").CreateFrom<SalaryData>(useHeader: true, separator: ','),
                new ColumnConcatenator("Features", "YearsExperience"),
                new GeneralizedAdditiveModelRegressor()
            };

            Console.WriteLine("--------------Training----------------");
            var model = pipeline.Train<SalaryData, SalaryPrediction>();

            // Evaluate
            Console.WriteLine(Environment.NewLine);
            Console.WriteLine("--------------Evaluating----------------");
            var testData = new TextLoader("SalaryData-test.csv").CreateFrom<SalaryData>(useHeader: true, separator: ',');

            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Root Mean Squared: {metrics.Rms}");
            Console.WriteLine($"R^2: {metrics.RSquared}");

            // Predict
            Console.WriteLine(Environment.NewLine);
            Console.WriteLine("--------------Predicting----------------");
            var prediction = model.Predict(new SalaryData { YearsExperience = PREDICTION_YEARS });

            Console.WriteLine($"After {PREDICTION_YEARS} years you would earn around {String.Format("{0:C}", prediction.PredictedSalary)}");

            Console.ReadLine();
        }
    }
}
