using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Math;
using AI.ML.ANN;

namespace AI.ML.Core
{
    public class Matrix
    {
        public static double[,] Add(double[,] x, double[,] y)
        {
            int r = x.GetLength(0), c = x.GetLength(1);
            double[,] s = new double[r, c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    s[i, j] = x[i, j] + y[i, j];
            return s;
        }

        public static void AddAt(double[][] x, double[][] y)
        {
            int r = x.Length, c = x[0].Length;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    y[i][j] += x[i][j];
        }

        public static void Add(double[,] x, double[,] y, double[,] _output)
        {
            int r = x.GetLength(0), c = x.GetLength(1);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    _output[i, j] = x[i, j] + y[i, j];
        }

        public static bool AreEqual(double[,] x, double[,] y) => (x.GetLength(0) == y.GetLength(0)) &&
            (x.GetLength(1) == y.GetLength(1));

        public static double[,] ConcatCols(double[,] a, double[,] b)
        {
            int r = a.GetLength(0);
            int u = a.GetLength(1);
            int v = b.GetLength(1);

            double[,] c = new double[r, u + v];

            for (int i = 0; i < r; i++)
                for (int j = 0; j < u; j++)
                    c[i, j] = a[i, j];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < v; j++)
                    c[i, j + u] = b[i, j];

            return c;
        }

        public static double[,] ConcatRows(double[,] a, double[,] b)
        {
            int c = a.GetLength(1);
            int u = a.GetLength(0);
            int v = b.GetLength(0);

            double[,] d = new double[u + v, c];

            for (int i = 0; i < u; i++)
                for (int j = 0; j < c; j++)
                    d[i, j] = a[i, j];
            for (int i = 0; i < v; i++)
                for (int j = 0; j < c; j++)
                    d[i + u, j] = b[i, j];

            return d;
        }

        public static double[][] Divide(double[][] x, int value)
        {
            int r = x.Length, c = x[0].Length;
            double[][] y = new double[r][];
            for (int i = 0; i < r; i++)
                y[i] = new double[c];

            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    y[i][j] = x[i][j] / value;
            return y;
        }

        public static double[][] Divide(double[][] x, double[][] z)
        {
            int r = x.Length, c = x[0].Length;
            double[][] y = new double[r][];
            for (int i = 0; i < r; i++)
                y[i] = new double[c];

            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    y[i][j] = x[i][j]/z[i][j];
            return y;
        }

        public static double[][] Pow(double[][] a, int pow)
        {
            int r = a.Length, c = a[0].Length;
            double[][] y = new double[r][];
            for (int i = 0; i < r; i++)
                y[i] = new double[c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    y[i][j] = System.Math.Pow(a[i][j], pow);
                }
            }

            return y;
        }

        public static double[][] Sqrt(double[][] a)
        {
            int r = a.Length, c = a[0].Length;
            double[][] y = new double[r][];
            for (int i = 0; i < r; i++)
                y[i] = new double[c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    y[i][j] = System.Math.Sqrt(a[i][j]);
                }
            }
            return y;
        }

        public static void Add(double[][] a, double value)
        {
            int r = a.Length, c = a[0].Length;
           
            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    a[i][j] += value;
                }
            }
        }

        public static double[][] AddValue(double[][] a, double value)
        {
            int r = a.Length, c = a[0].Length;
            double[][] z = new double[r][];
            for (int i = 0; i < r; i++)
                z[i] = new double[c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    z[i][j] = a[i][j] + value;
                }
            }
            return z;
        }

        public static double[,] Constant(double x, int row, int col)
        {
            double[,] a = new double[row, col];

            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    a[i, j] = x;

            return a;
        }

        /// <summary>
        /// make copy of matrix
        /// </summary>
        /// <param name="m">matrix</param>
        /// <returns>copy of matrix</returns>
        public static double[,] Copy(double[,] m)
        {
            int r = m.GetLength(0), c = m.GetLength(1);
            double[,] x = new double[r, c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    x[i, j] = m[i, j];
            return x;
        }

        public static double[,] Copy(double[,] matrix, int r, int c, int rows, int cols)
        {
            double[,] m = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = matrix[r + i, c + j];
            return m;
        }

        public static double[,] Copy(double[,,] matrix, int n, int r, int c, int rows, int cols)
        {
            double[,] m = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = matrix[n, r + i, c + j];
            return m;
        }

        /// <summary>
        /// inserts source into target
        /// </summary>
        /// <param name="x">source</param>
        /// <param name="r_offset">row offset</param>
        /// <param name="c_offset">col offset</param>
        /// <param name="y">target</param>
        /// <returns></returns>
        public static void InsertAt(double[,] x, double[,] y, int r_offset, int c_offset)
        {
            // 0. insert
            int r = x.GetLength(0);
            int c = x.GetLength(1);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    y[i + r_offset, j + c_offset] = x[i, j];
        }

        public static void InsertAt(double[,] matrix, double[,,] target, int n_offset, int r_offset, int c_offset)
        {
            int r = matrix.GetLength(0), c = matrix.GetLength(1);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    target[n_offset, r_offset + i, c_offset + j] = matrix[i, j];
        }

        public static double[,] Logt(double[,] x, params double[] z)
        {
            int r = x.GetLength(0);
            int c = x.GetLength(1);

            double u;
            double[,] y = new double[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    u = 1.0 + System.Math.Exp(-1.0 * z[1] * x[i, j]);
                    y[i, j] = z[0] / u;
                }
            }

            return y;
        }

        public static void Logt(double[,] _source, double[,] _target, params double[] _logt)
        {
            int r = _source.GetLength(0);
            int c = _source.GetLength(1);

            double u;

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    u = 1.0 + System.Math.Exp(-1.0 * _logt[1] * _source[i, j]);
                    _target[i, j] = _logt[0] / u;
                }
            }
        }

        public static double[][] Multiply(double[][] a, double b)
        {
            int r = a.Length, c = a[0].Length;
            double[][] y = new double[r][];
            for (int i = 0; i < r; i++)
                y[i] = new double[c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    y[i][j] = a[i][j] * b;
                }
            }
            return y;
        }

        public static double[,] Multiply(double[,] a, double[,] b)
        {
            int r = a.GetLength(0), c = b.GetLength(1), x = a.GetLength(1);

            double[,] p = new double[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    p[i, j] = 0.0;
                    for (int k = 0; k < x; k++)
                        p[i, j] += a[i, k] * b[k, j];
                }
            }

            return p;
        }

        public static double[,] Multiply(double[,,] a, int s, double[,] b)
        {
            int r = a.GetLength(1), c = b.GetLength(1), x = b.GetLength(0);

            double[,] p = new double[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    p[i, j] = 0.0;
                    for (int k = 0; k < x; k++)
                        p[i, j] += a[s, i, k] * b[k, j];
                }
            }

            return p;
        }

        public static double[,] Multiply(double[,] a, List<double[,]> l)
        {
            int r = a.GetLength(0), c = l[0].GetLength(1), x = a.GetLength(1), u;

            double[,] t;

            double[,] p = new double[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    p[i, j] = 0.0;
                    for (int m = 0, k = 0; k < x; m++)
                    {
                        t = l[m];
                        u = t.GetLength(0);
                        for (int n = 0; n < u; n++)
                        {
                            p[i, j] += a[i, k] * t[n, j];
                            ++k;
                        }
                    }
                }
            }

            return p;
        }

        public static double[,] Multiply(double[,,] a, int s, List<double[,]> l)
        {
            int x = 0;
            for (int i = 0; i < l.Count; i++)
                x = l[i].GetLength(0);

            int r = a.GetLength(1), c = l[0].GetLength(1), u;

            double[,] t;

            double[,] p = new double[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    p[i, j] = 0.0;
                    for (int m = 0, k = 0; k < x; m++)
                    {
                        t = l[m];
                        u = t.GetLength(0);
                        for (int n = 0; n < u; n++)
                        {
                            p[i, j] += a[s, i, k] * t[n, j];
                            ++k;
                        }
                    }
                }
            }

            return p;
        }

        /// <summary>
        /// element wise multiplication
        /// </summary>
        /// <param name="x">matrix</param>
        /// <param name="y">matrix</param>
        /// <returns></returns>
        public static double[,] MultiplyByElements(double[,] x, double[,] y)
        {
            int r = x.GetLength(0);
            int c = x.GetLength(1);

            double[,] z = new double[r, c];

            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    z[i, j] = x[i, j] * y[i, j];

            return z;
        }

        public static double[][] MultiplyByElements(double[][] x, double[][] y)
        {
            int r = x.Length;
            int c = x[0].Length;

            double[][] z = new double[r][];
            for (int i = 0; i < r; i++)
                z[i] = new double[c];

                for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    z[i][j] = x[i][j] * y[i][j];
            return z;
        }

        /// <summary>
        /// element wise multiplication for vectors
        /// </summary>
        /// <param name="x">vector</param>
        /// <param name="y">vector</param>
        /// <returns></returns>
        public static double[] MultiplyByElements(double[] x, double[] y)
        {
            if (x.Length != y.Length)
                throw new Exception("Size mismatch");

            int r = x.Length;

            double[] z = new double[r];

            for (int i = 0; i < r; i++)                
                z[i] = x[i] * y[i];

            return z;
        }


        /// <summary>
        /// fill with random numbers with normal distribution
        /// </summary>
        /// <param name="x">matrix</param>
        /// <param name="m">mean</param>
        /// <param name="v">variance</param>
        public static void RandomNormal(double[,] x, double m, double v)
        {
            int r = x.GetLength(0), c = x.GetLength(1);
            //
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    x[i, j] = Math.Daemon.NextGaussianDouble(m, v)[0];
        }

        public static void RandomNormal(double[][] x, double m, double v)
        {
            int r = x.Length, c = x[0].Length;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                {
                    x[i][j] = Math.Daemon.NextGaussianDouble(m, v)[0];
                }
        }

        public static void RandomStandardNormal(double[][] x, double scale)
        {
            int r = x.Length, c = x[0].Length;
            Random rand = new Random();
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                {
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    double randStdNormal = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Sin(2.0 * System.Math.PI * u2); //random normal(0,1)
                    x[i][j] = randStdNormal * scale;
                }
        }

        public static void RandomNormal(double[][] x, int mean, int stddev, double scale)
        {
            int r = x.Length, c = x[0].Length;
            Random rand = new Random();
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                {
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    double randStdNormal = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Sin(2.0 * System.Math.PI * u2); //random normal(0,1)
                    double randNormal = mean + stddev * randStdNormal;
                    x[i][j] = randNormal * scale;
                }
        }
        public static double[] MeanVariance(double[][] x)
        {
            int r = x.Length, c = x[0].Length;
            Random rand = new Random();
            double sum = 0.0;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                {
                    sum += x[i][j];
                }
            double mean = sum / (r * c);

            sum = 0.0;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                {
                    sum += System.Math.Pow(x[i][j] - mean, 2);
                }
            double variance = sum / (r * c);
            return new double[] { mean, variance};
        }


        /// <summary>
        /// fill with random numbers with normal distribution
        /// </summary>
        /// <param name="x">2-dimensional matrix</param>
        /// <param name="m">mean</param>
        /// <param name="v">variance</param>
        public static void RandomNormal(double[,,] x, double m, double v)
        {
            int n = x.GetLength(0), r = x.GetLength(1), c = x.GetLength(2);
            //
            for (int i = 0; i < n; i++)
                for (int j = 0; j < r; j++)
                    for (int k = 0; k < c; k++)
                        x[i, j, k] = Math.Daemon.NextGaussianDouble(m, v)[0];
        }

        /// <summary>
        /// fill with random numbers with normal distribution
        /// </summary>
        /// <param name="x">3-dimensional matrix</param>
        /// <param name="m">mean</param>
        /// <param name="v">variance</param>
        public static void RandomNormal(double[,,] x, int n, double m, double v)
        {
            int r = x.GetLength(0), c = x.GetLength(1);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    x[n, i, j] = Math.Daemon.NextGaussianDouble(m, v)[0];
        }

        /// <summary>
        /// fill maxtrix with random numbers with uniform distribution
        /// </summary>
        /// <param name="x">matrix</param>
        /// <param name="min">lower bound</param>
        /// <param name="max">upper bound</param>
        public static void RandomUniform(double[,] x, double min, double max)
        {
            int r = x.GetLength(0), c = x.GetLength(1);
            double d = max - min;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    x[i, j] = min + Math.Daemon.Random.NextDouble() * d;
        }
        public static void RandomUniform(double[,,] x, double min, double max)
        {
            int n = x.GetLength(0), r = x.GetLength(1), c = x.GetLength(2);
            double d = max - min;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < r; j++)
                    for (int k = 0; k < c; k++)
                        x[i, j, k] = min + Math.Daemon.Random.NextDouble() * d;
        }

        /// <summary>
        /// fill maxtrix with random numbers with uniform distribution
        /// </summary>
        /// <param name="x">matrix</param>
        /// <param name="min">lower bound</param>
        /// <param name="max">upper bound</param>
        public static void RandomUniform(double[,,] x, int n, double min, double max)
        {
            int r = x.GetLength(0), c = x.GetLength(1);
            double d = max - min;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    x[n, i, j] = min + Math.Daemon.Random.NextDouble() * d;
        }

        public static double[,] Subtract(double[,] x, double[,] y)
        {
            int r = x.GetLength(0), c = x.GetLength(1);
            double[,] s = new double[r, c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    s[i, j] = x[i, j] - y[i, j];
            return s;
        }

        public static void Subtract(double[,] x, double[,] y, double[,] _output)
        {
            int r = x.GetLength(0), c = x.GetLength(1);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    _output[i, j] = x[i, j] - y[i, j];
        }

        /// <summary>
        /// tanh activation function
        /// </summary>
        /// <param name="_source">source</param>
        /// <param name="_tanh">function parameters</param>
        /// <returns></returns>
        public static double[,] Tanh(double[,] _source, params double[] _tanh)
        {
            int r = _source.GetLength(0);
            int c = _source.GetLength(1);

            double u; double[,] y = new double[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    u = _tanh[1] * _source[i, j];
                    y[i, j] = _tanh[0] * System.Math.Tanh(u);
                }
            }

            return y;
        }

        /// <summary>
        /// tanh activation function
        /// </summary>
        /// <param name="_source">source</param>
        /// <param name="_output">target</param>
        /// <param name="_tanh">function parameters</param>
        public static void Tanh(double[,] _source, double[,] _output, params double[] _tanh)
        {
            int r = _source.GetLength(0);
            int c = _source.GetLength(1);

            double u;

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    u = _tanh[1] * _source[i, j];
                    _output[i, j] = _tanh[0] * System.Math.Tanh(u);
                }
            }
        }

        /// <summary>
        /// print matrix
        /// </summary>
        /// <param name="x">2-d matrix</param>
        /// <returns></returns>
        public static string ToString(double[,] x)
        {
            string s = ""; char[] t = new char[] { ' ' };

            int r = x.GetLength(0);
            int c = x.GetLength(1);

            for (int i = 0; i < r; i++)
            {
                s += "\n[";
                for (int j = 0; j < c; j++)
                    s += Math.Method.Sign(x[i, j]) + x[i, j].ToString(Global.Precision) + " ";
                s = s.TrimEnd(t) + "]";
            }

            return s;
        }
        public static string ToString(double[][] x)
        {
            string s = ""; char[] t = new char[] { ' ' };

            int r = x.Length;
            int c = x[0].Length;

            for (int i = 0; i < r; i++)
            {
                s += "\n[";
                for (int j = 0; j < c; j++)
                    s += Math.Method.Sign(x[i][j]) + x[i][j].ToString(Global.Precision) + " ";
                s = s.TrimEnd(t) + "]";
            }

            return s;
        }

        /// <summary>
        /// print matrix
        /// </summary>
        /// <param name="x">3-d matrix</param>
        /// <returns></returns>
        public static string ToString(double[,,] x)
        {
            string s = ""; char[] t = new char[] { ' ' };

            int n = x.GetLength(0);
            int r = x.GetLength(1);
            int c = x.GetLength(2);

            for (int i = 0; i < n; i++)
            {
                s += "\n[n => " + i.ToString("00") + "]";
                for (int j = 0; j < r; j++)
                {
                    s += "\n[";
                    for (int k = 0; k < c; k++)
                        s += Math.Method.Sign(x[i, j, k]) + x[i, j, k].ToString(Global.Precision) + " ";
                    s = s.TrimEnd(t) + "]";
                }
            }

            return s;
        }

        public static double[,] Transpose(double[,] x)
        {
            int r = x.GetLength(0);
            int c = x.GetLength(1);

            double[,] t = new double[c, r];
            for (int i = 0; i < c; i++)
                for (int j = 0; j < r; j++)
                    t[i, j] = x[j, i];
            return t;
        }

        public static double[][] Transpose(double[][] x)
        {
            int r = x.Length;
            int c = x[0].Length;

            double[][] t = new double[c][];
            for (int i = 0; i < c; i++)
                t[i] = new double[r];

            for (int i = 0; i < c; i++)
                for (int j = 0; j < r; j++)
                    t[i][j] = x[j][i];
            return t;
        }

        public static double[,] Unit(int n)
        {
            double[,] u = new double[n, n];

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    u[i, j] = i == j ? 1.0 : 0.0;

            return u;
        }

        public static double[,] Zero(int r, int c)
        {
            double[,] z = new double[r, c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    z[i, j] = 0.0;
            return z;
        }

        public static void Zero(double[][] x)
        {            
            for (int i = 0; i < x.Length; i++)
                for (int j = 0; j < x[0].Length; j++)
                    x[i][j] = 0.0;
        }

        /// <summary>
        /// constructs a zero matrix
        /// </summary>
        /// <param name="n">segments</param>
        /// <param name="r">rows</param>
        /// <param name="c">cols</param>
        /// <returns></returns>
        public static double[,,] Zero(int n, int r, int c)
        {
            double[,,] z = new double[n, r, c];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < r; j++)
                    for (int k = 0; k < c; k++)
                        z[i, j, k] = 0.0;
            return z;
        }


        /// <summary>
        /// flatten a 2d matrix
        /// </summary>
        /// <param name="x">2d matrix</param>        
        /// <returns></returns>
        public static double[] Flatten(double[,] x)
        {
            int r = x.GetLength(0);
            int c = x.GetLength(1);
            int l = 0;
            double[] z = new double[r * c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                {
                    z[l] = x[i, j];
                    l++;
                }
            return z;
        }

        public static double[] Flatten(double[][] x)
        {
            int r = x.Length;
            int c = x[0].Length;
            int l = 0;
            double[] z = new double[r * c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                {
                    z[l] = x[i][j];
                    l++;
                }
            return z;
        }

        /// <summary>
        /// flatten a 3d matrix
        /// </summary>
        /// <param name="x">3d matrix</param>        
        /// <returns></returns>
        public static double[] Flatten(double[,,] x)
        {
            int n = x.GetLength(0);
            int r = x.GetLength(1);
            int c = x.GetLength(2);
            int l = 0;
            double[] z = new double[n * r * c];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < r; j++)
                    for (int k = 0; k < c; k++)
                    {
                        z[l] = x[i, j, k];
                        l++;
                    }
            return z;
        }

        internal static double[] Flatten(double?[][] x)
        {
            int r = x.Length;
            int c = x[0].Length;
            int l = 0;
            double[] z = new double[r * c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                {
                    z[l] = x[i][j].Value;
                    l++;
                }
            return z;
        }        
    }
}