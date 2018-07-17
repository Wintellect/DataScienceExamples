using System;
using Microsoft.ML;
using MLNetShared;

namespace LoadModel
{
    class Program
    {
        private static int YEARS_EXPERIENCE = 8;

        static void Main(string[] args)
        {
            var modelPath = MLNetUtilities.GetModelFilePath("model.zip");

            var model = PredictionModel.ReadAsync<SalaryData, SalaryPrediction>(modelPath).Result;

            var prediction = model.Predict(new SalaryData { YearsExperience = YEARS_EXPERIENCE });

            Console.WriteLine($"Prediction for {YEARS_EXPERIENCE} years - {prediction.PredictedSalary}");

            // Can write it back to the file system
            // model.WriteAsync("model.zip");

            Console.ReadLine();
        }
    }
}
