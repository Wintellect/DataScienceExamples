using Microsoft.ML.Runtime.Api;

namespace MLNetShared
{
    public class SalaryData
    {
        [Column("0")]
        public float YearsExperience;

        [Column("1", name: "Label")]
        public float Salary;
    }
}
