using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MLNetShared;
using System;
using System.Linq;

namespace CrossValidation
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataset = MLNetUtilities.GetDataPathByDatasetName("SalaryData.csv");
            var testDataset = MLNetUtilities.GetDataPathByDatasetName("SalaryData-test.csv");

            var pipeline = new LearningPipeline
            {
                new TextLoader(dataset).CreateFrom<SalaryData>(useHeader: true, separator: ','),
                new ColumnConcatenator("Features", "YearsExperience"),
                new GeneralizedAdditiveModelRegressor()
            };

            var crossValidator = new CrossValidator() { Kind = MacroUtilsTrainerKinds.SignatureRegressorTrainer, NumFolds = 10 };
            var crossValidatorOutput = crossValidator.CrossValidate<SalaryData, SalaryPrediction>(pipeline);

            var r2 = crossValidatorOutput.RegressionMetrics.Sum(metric => metric.RSquared);

            var totalR2 = crossValidatorOutput.RegressionMetrics.Sum(metric => metric.RSquared);
            var totalRMS = crossValidatorOutput.RegressionMetrics.Sum(metric => metric.Rms);

            Console.WriteLine($"Average R^2: {totalR2 / crossValidatorOutput.RegressionMetrics.Count}");
            Console.WriteLine($"Average RMS: {totalRMS / crossValidatorOutput.RegressionMetrics.Count}");

            Console.ReadLine();
        }
    }
}
