# Set directory
setwd("/Users/samueleravazzani/Library/Mobile Documents/com~apple~CloudDocs/Politecnico di Milano/2 anno M/1 semestre/E-health Methods and Applications/Project/Projectpython")

# Create your contingency table (replace with your actual data)
data <- matrix(c(28,0,3,10,24,31,7,13,6,0,5,3,0,8,0,6,2,4), nrow = 6, byrow = TRUE)
print(data)

# Perform Fisher's exact test
result <- fisher.test(data, workspace = 2e8)

# Print the result
print(result)

## sfortunatamente escono che ci sono differenze

# 0 - 1
matrix_0_1 <- data[,c(1,2)]
print(matrix_0_1)
result_0_1 <- fisher.test(matrix_0_1, workspace = 2e8)
print(result_0_1)

# 0 - 2
matrix_0_2 <- data[,c(1,3)]
print(matrix_0_2)
result_0_2 <- fisher.test(matrix_0_2, workspace = 2e8)
print(result_0_2)

# 1 - 2
matrix_1_2 <- data[,c(2,3)]
print(matrix_1_2)
result_1_2 <- fisher.test(matrix_1_2, workspace = 2e8)
print(result_1_2)