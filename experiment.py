from bitflip import train




if __name__ == "__main__":
    bits = 25
    epochs = 150

    for her in [True, False]:
        success = train(bits, epochs, her)
        plt.plot(success, label="HER-DQN" if her else "DQN")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.title("Number of bits: {}".format(bits))
    plt.savefig("{}_bits.png".format(bits), dpi=1000)
    plt.show()