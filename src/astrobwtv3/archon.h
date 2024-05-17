typedef	unsigned int	suffix;
typedef	unsigned int	t_index;
typedef unsigned char	byte;
typedef unsigned short	dbyte;
typedef unsigned long	word;


class Archon	{
	const t_index Nmax, Nreserve;
	suffix *const P;
	byte *const str;
	t_index N, baseId;
	void roll(const t_index i);

public:
	static t_index estimateReserve(const t_index);
	Archon(const t_index N);
	~Archon();
	unsigned countMemory() const;
	bool validate();
	// encoding
	int enRead(FILE *const fx, t_index ns);
	int enCompute();
	int enWrite(FILE *const fx);
	// decoding
	int deRead(FILE *const fx, t_index ns);
	int deCompute();
	int deWrite(FILE *const fx);
};