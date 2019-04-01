from matplotlib import pyplot as plt
def data_explore(df):
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.figure(1)
    plt.subplot(221)
    plt.title("Amount")
    df.amount.hist()

    plt.subplot(222)
    plt.title("Amount < 300")
    df[df.amount<300].amount.hist(bins=10)
    
    plt.subplot(223)
    plt.title("Fraud distribution")
    df.fraud.hist()

    df_fraud_only = df[df["fraud"]==1]
    
    plt.subplot(224)
    plt.title("Fraud amount distribution")
    df_fraud_only[df_fraud_only.amount<9000].amount.hist(bins=10)

    return(plt)
