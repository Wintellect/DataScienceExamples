using Microsoft.Cognitive.CustomVision.Prediction;
using Microsoft.Cognitive.CustomVision.Training;
using Microsoft.Cognitive.CustomVision.Training.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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

            Console.WriteLine("Press 1 to predict and 2 to train:");
            var pathChoice = Console.ReadLine();

            if ("1".Equals(pathChoice))
            {
                Console.WriteLine("Input path to image to test:");
                var imagePath = Console.ReadLine();

                if (!File.Exists(imagePath))
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
            else
            {
                Console.WriteLine("Input path to image to train model with:");
                var imagePath = Console.ReadLine();

                Console.WriteLine("What tag would you give this image? Rosemary, cilantro, or basil?");
                var imageTag = Console.ReadLine();

                var capitilizedTag = char.ToUpper(imageTag.First()) + imageTag.Substring(1).ToLower();

                if (!File.Exists(imagePath))
                {
                    Console.WriteLine("File does not exist. Press enter to exit.");
                    Console.ReadLine();
                    return;
                }

                var imageFile = File.OpenRead(imagePath);

                var tags = trainingApi.GetTags(herbProject.Id);

                var matchedTag = tags.Tags.FirstOrDefault(t => t.Name == capitilizedTag);

                var memoryStream = new MemoryStream();
                imageFile.CopyTo(memoryStream);

                var fileCreateEntry = new ImageFileCreateEntry(imageFile.Name, memoryStream.ToArray());
                var fileCreateBatch = new ImageFileCreateBatch { Images = new List<ImageFileCreateEntry> { fileCreateEntry }, TagIds = new List<Guid> { matchedTag.Id } };

                var result = trainingApi.CreateImagesFromFiles(herbProject.Id, fileCreateBatch);

                var resultImage = result.Images.FirstOrDefault();

                switch(resultImage.Status)
                {
                    case "OKDuplicate":
                        Console.WriteLine("Image is already used for training. Please use another to train with");
                        Console.ReadLine();
                        break;
                    default:
                        break;
                }

                var iteration = trainingApi.TrainProject(herbProject.Id);

                while (iteration.Status != "Completed")
                {
                    System.Threading.Thread.Sleep(1000);

                    iteration = trainingApi.GetIteration(herbProject.Id, iteration.Id);
                }

                iteration.IsDefault = true;
                trainingApi.UpdateIteration(herbProject.Id, iteration.Id, iteration);
                Console.WriteLine("Done training!");

                Console.ReadLine();
            }
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
