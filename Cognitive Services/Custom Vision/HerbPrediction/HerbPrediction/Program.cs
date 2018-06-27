using Microsoft.Cognitive.CustomVision.Prediction;
using Microsoft.Cognitive.CustomVision.Training;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HerbPrediction
{
    class Program
    {
        static void Main(string[] args)
        {
            Keys keys = null;

            using (var reader = new StreamReader("App.json"))
            {
                var json = reader.ReadToEnd();

                keys = JsonConvert.DeserializeObject<Keys>(json);
            }

            var trainingApi = new TrainingApi { ApiKey = keys.TrainingKey };
            var predictionEndpoint = new PredictionEndpoint { ApiKey = keys.PredictionKey };

            var projects = trainingApi.GetProjects();
            var herbProject = projects.Where(p => p.Name == "Herbs").FirstOrDefault();

            Console.WriteLine("Predicting basil image");
            var imageFile = File.OpenRead("basil_test.jpg");

            if (herbProject != null)
            {

                var result = predictionEndpoint.PredictImage(herbProject.Id, imageFile);

                foreach (var prediction in result.Predictions)
                {
                    Console.WriteLine($"Tag: {prediction.Tag} Probability: {prediction.Probability}");
                }
            }
            else
            {
                Console.WriteLine("Project doesn't exist.");
            }

            Console.ReadLine();
        }
    }
}
