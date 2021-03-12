using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;


namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            Network network = new Network();

            int epoch = 300;
            Console.ReadLine();

            for (int i = 0; i < epoch; i++)
            {
                network.forward();
                Console.WriteLine("Epoch: " + i + " Total Error: " + network.Etotal);
                network.backward();
                network.weight_adjustment();
                
            }

            Console.ReadLine();
        }
    }
}
