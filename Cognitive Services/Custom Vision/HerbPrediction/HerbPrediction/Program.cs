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
            var keys = GetApiKeys();

            var trainingApi = new TrainingApi { ApiKey = keys.TrainingKey };
            var predictionEndpoint = new PredictionEndpoint { ApiKey = keys.PredictionKey };

            var projects = trainingApi.GetProjects();
            var herbProject = projects.Where(p => p.Name == "Herbs").FirstOrDefault();

            Console.WriteLine("Input path to image to test:");
            var imagePath = Console.ReadLine();

            if(!File.Exists(imagePath))
            {
                Console.WriteLine("File does not exist. Press enter to exit.");
                Console.ReadLine();
                return;
            }

            Console.WriteLine("Image predictions:");

            var imageFile = File.OpenRead(imagePath);

            if (herbProject != null)
            {
                var result = predictionEndpoint.PredictImage(herbProject.Id, imageFile);

                foreach (var prediction in result.Predictions)
                {
                    Console.WriteLine($"Tag: {prediction.Tag} Probability: {String.Format("Value: {0:P2}.", prediction.Probability)}");
                }
            }
            else
            {
                Console.WriteLine("Project doesn't exist.");
            }

            Console.ReadLine();
        }

        private static Keys GetApiKeys()
        {
            Keys keys = null;

            using (var reader = new StreamReader("App.json"))
            {
                var json = reader.ReadToEnd();

                keys = JsonConvert.DeserializeObject<Keys>(json);
            }

            return keys;
        }
    }
}
