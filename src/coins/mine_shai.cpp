#include "miners.hpp"
#include "tnn-hugepages.h"
#include "hex.h"

#include <crypto/shai/shai-hive.h>

std::string convertPathToHexString(const std::vector<uint16_t> &path)
{
  std::ostringstream oss;
  for (const auto &val : path)
  {
    // Convert to little-endian bytes
    uint8_t byte1 = static_cast<uint8_t>(val & 0xFF);        // Least significant byte
    uint8_t byte2 = static_cast<uint8_t>((val >> 8) & 0xFF); // Most significant byte

    // Format as hexadecimal
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byte1);
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byte2);
  }
  return oss.str();
}

std::string convertPathToHexString(uint16_t *path)
{
  std::ostringstream oss;
  for (int i = 0; i < 2008; i++)
  {
    uint16_t val = path[i];
    // Convert to little-endian bytes
    uint8_t byte1 = static_cast<uint8_t>(val & 0xFF);        // Least significant byte
    uint8_t byte2 = static_cast<uint8_t>((val >> 8) & 0xFF); // Most significant byte

    // Format as hexadecimal
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byte1);
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byte2);
  }
  return oss.str();
}

std::string byteArrayToHexString(const uint8_t *byteArray, size_t length)
{
  std::ostringstream oss;

  for (size_t i = 0; i < length; ++i)
  {
    // Format each byte as a two-digit hex value
    oss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byteArray[i]);
  }

  return oss.str();
}

bool meets_target(std::string hash, std::string target)
{
  Num target_int = Num(target.c_str(), 16); //.expect("Invalid target hex string");
  Num hash_int = Num(hash.c_str(), 16);     //.expect("Invalid hash hex string");
  // std::cout << hash_int << " < " << target_int << std::endl << std::flush;
  return hash_int < target_int;
}

uint32_t getLeastSignificant32Bits(uint64_t value)
{
  return static_cast<uint32_t>(value & 0xFFFFFFFF);
}

void mineShai(int tid)
{
  byte random_buf[12];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  std::array<int, 12> buf;
  std::generate(buf.begin(), buf.end(), [&dist, &gen]()
                { return dist(gen); });
  std::memcpy(random_buf, buf.data(), buf.size());

  boost::this_thread::sleep_for(boost::chrono::milliseconds(125));

  int64_t localJobCounter;
  byte powHash[10000];
  byte work[ShaiHive::SHAI_DATA_SIZE];
  byte devWork[ShaiHive::SHAI_DATA_SIZE];

  ShaiHive::ShaiCtx workCtx;

  std::string expected_input("00000020cafc0ab25e5afc8bfc42afe37d22b704c01e68003966d53498816d7204000000c52d2387493e40bfe688d366200e6cf1ab7ee8b668480306ad0afb214eced8b62a2b2867f35e061daecd29e8");
  std::string expected_hash("000d2b4c2d0181c1418237837c825481f5690ef36f01455528ce6b1b487a6c3a");
  std::string expected_path_hex("000001000300020004000500070009000b00060008000a000c000f000e000d0010001100120014001600130017001e001500190018001a0020001d001c001f0022001b00250023002b002100240028002600290027002c0030002a0031003300320034002e002d002f0035003600380037003a003c003f003b0039003d00400042003e0041004300440047004800450049004d0046004b004c004a004e0053004f00570054005000520051005600580059005b0060005c005d0055005e00650061005a005f0064006200660063006e006900680067006d006a006c006b00720070006f0078007100740073007500760079007a007b0077007c007f0080007d0082007e0085008300840081008700860088008a0089008d0090008c008b0091008e008f00920095009800940096009300970099009b009a009e009f009c009d00a000a200a300a100a400a700a500a900a800a600ab00ac00aa00ae00ad00af00b000b100b200b300b400b500b600bb00ba00b700bc00be00b900bd00bf00b800c000c200c100c500c800c300c400c600c700cb00c900ca00cc00cd00ce00cf00d100d000d200d500d300d600d800d700d400d900da00db00dd00dc00e100e500de00e300df00e700e000e200e800e400e600e900eb00ea00ec00ee00ed00ef00f300f400f000f500f100f600fb00f200f900f700f800fa00fc00fd00fe00020101010301050106010401ff000801090100010a01110107010b010d010c010e010f01100112011301160114011901170118011a011b011c0115011d011f0121011e012401200126012201250129012b0123012a0127012c0128012d012f012e013101300132013401360137013301350138013a0139013d013b013e013c013f0140014101420143014401460147014501480149014c014a014b014d014f01510150014e015301540155015a015601520157015b01590158015d015c0164015f0160015e0161016201660165016a0168016301670169016b016c016e016d016f01710170017301720174017501770178017c01760179017a017b017f017e0180017d018101850182018401880189018301870186018a018e018b018c018d01900191018f01920195019301960194019a019c019701990198019b019e019f01a001a101a301a501a401a6019d01a801a201ad01a901a701ae01aa01b001b201ab01ac01b401b601b101b501b901bb01b301af01b801b701bd01be01bf01c001bc01c201ba01c101c301c401c601c701c501cc01c801cd01c901cb01cf01ca01d001d101d201d301d401d501ce01d801d701d601da01d901db01dc01de01e101dd01e001df01e301e201e401e501e701e601e801e901ea01eb01ed01ee01f101ec01ef01f001f301f201f801f401f601f501f701f901fa01fe01fb01ff01fc010302fd010102000204020602020205020702080209020a020b020d020e0212020f020c02100211021302140217021502180216021b0219021a021f021c021e0221021d02220226022002230225022402270228022b0229022a022e022f022d022c0231023302320234023002390235023702360238023d023a023b023c023f023e024002420241024802430249024402470246024a0245024c024e024f0253024b024d025102520256025402580257025a025902550250025c025b025d0260025f025e02620264026d0265026102630267026902680266026a026f026c026e027102700273027202740277026b027502790278027c027b027a0276027d0280027e027f0281028202870284028302850289028a02880286028d028b028e0290028f0291028c0293029202950294029a02960298029b0299029c029e029d02a0029702a102a302a202a7029f02a602a402a902a802a502ab02ac02aa02ad02af02b002b302b202ae02b102b702b402b502b602b902ba02b802bb02bc02c102c302c402bf02bd02be02c702c002c902c502c202c602c802cc02ca02d002cd02cf02d202d902d102cb02ce02d402d302d602d502d802da02df02d702dc02db02dd02e002e402e102e502e602de02e302e202e902e702e802eb02ea02ec02ed02f502f102ee02ef02f002f602f202f302f702f402f802f902fa02fd02fb02fc02fe02ff02010300030203030304030703060305030a03090308030c030b030d030f03100311030e03120313031403170316031903180315031a031c031b031f031e032003220323031d0325032403260327032903280321032a032b032d032c0331032f032e0333033003360335033403370332033d0339033a0338033c033e033b033f03400341034403420345034703430348034a0346034b034e034d0349034c0350034f0351035603520353035503540357035a0358035c035b0359035d035e0362035f03650367036003640361036303680369036a0366036c036d036b036f036e037103720370037303740375037a0378037603770379037d037e0381037b037c037f03820383038603800384038503890387038b038a038c03880391038e038d0390038f03920396039303950394039a03970399039d039e039c039803a0039f039b03a103a203a803a403a303a503a603a903af03a703ad03aa03ac03b103ae03b003ab03b303b203b803b503b403b603b903bb03bc03b703bd03ba03be03bf03c003c103c303c203c403c503c803ca03c703c603c903cc03cb03d103cd03ce03d303cf03d003d503d603d403d703d203d803da03dd03d903dc03de03db03df03e103e003e203e303e403e503e703e603e803e903ef03ea03ec03eb03ed03f103ee03f003f403f203f303f503f703f603fa03f903f803fc03fb030004fd03ff03fe0303040404010402040504080406040b0407040d040a0409040c040e040f04110413041204100418041404150416041b0417041a041c041d0419041e041f0420042204240421042304280427042504260429042a042b042c042e042d042f0430043104330435043604320434043804370439043b043d043c043e0440043f0441043a0446044204440443044904500447044a044d04480445044e044f044b044c04520453045504510456045804540457045b0459045d045a045c045f0463045e046504600462046104680464046a04690466046b0467046e0471046c046d0472046f0470047304740477047504760479047a047d047e047b047f04780480047c0481048204860489048404830485048a0488048c0487048b048f048d048e04900491049204930497049c0498049d0494049504990496049f04a0049a049b049e04a104a504a204a304a604a804aa04a404a704a904ab04ad04b204ac04b004ae04af04b304b504b104b604b404b904ba04be04b804b704bc04bb04c004bd04c104c204c504bf04c404c604c304ca04c704c804cd04c904cb04ce04cf04d004cc04d104d304d204d504d704d404d804da04db04d604dc04dd04df04de04d904e104e004e304e404e204e504e704e604e804ed04eb04e904ea04ef04ee04f104f204ec04f004f304f704f504fa04f604f904f804f404fb04fc04fd04fe040205ff04000503050105040505050605080507050b050a050c0509050d050e050f051105130519051405120516051505100517051a051c051b0518051f051e051d05210520052205230525052405270526052a05280529052d0530052b052e0531052c052f05320535053a05330536053e0534053c053d05380537053b05390540053f054105420543054705440545054605490548054b054a054d054c054e054f055005520551055305550554055705560558055a0559055e055b0563055c0560055d0562056105670564055f056505660569056a056b0568056d056e056c0570056f057205710573057405750578057605770579057a057b057c057d057f0582057e0580058105830584058505860588058a0587058e058b0589058c058f0592058d05900591059305940595059905960597059a0598059c05a2059b05a3059d05a1059f059e05a005a605a405a705a805a505ac05aa05a905ab05ae05ad05af05b005b105b305b205b605bd05b405b905b705b805b505bb05bf05ba05be05c005bc05c105c605c405c505c705c205c805ca05cd05c905c305cb05cc05d105d005d305cf05d205ce05d905d505d405d605d805dd05da05d705db05dc05de05e605df05e005e105e205e405e505e805e905e305e705ea05ec05ee05eb05f005ed05f105f705f505f205ef05f305f405f605f805fb05f905fa05fc050106fe05fd0502060306ff050006040605060806060607060b060a060c060e060d0609060f061306140612061106150610061706160618061b061c061e061d0619061a06200623061f06210622062406250626062706280629062e062a062b062c062d062f063006310634063506320633063606370639063a0638063b063c063d0640063f063e06440641064306420646064b064706480649064a0645064c064d064f064e0651065206540653065006550657065806560659065a065c065b065d065e06610660065f06640665066306660662066a066b06680669066c0667066d066e066f06710672067706730670067d0675067406790676067a067b0678067e067c067f06820681068306800684068506890686068a06870688068d068b068c068e068f06900692069106940693069506980696069d069a069706a00699069b069c06a2069f06a306a1069e06a406a506aa06a606a706a806a906ad06ab06ac06ae06af06b006b106b206b306b406b506b606b806b706b906ba06bb06bc06bd06be06bf06c006c306c106c606c506c206c706c806c406c906ca06cb06cd06cf06ce06cc06d206d306d006d406d106d506d606d706d906d806db06df06da06dc06e206dd06e306e006e506de06e406e606e106e706e906ea06e806ec06ed06f306ef06eb06ee06f006f406f106f606fa06f506f206f706f806fc06fb06fe06f906000703070107fd060407ff0602070607050707070a07080709070b070d070e0711070f07100712071707180713070c071407190715071a0716071b071d071c071f071e0721072007230722072507280726072707290724072f072b072c072a0730072e0733072d07310732073407350738073607370739073a073c073b073d073f073e07400741074307440742074507460747074f0749074c074e074b0748074a0750074d075307510754075507520756075c07570759075b0758075d075a075e07600763075f076107660762076707640765076a07680769076c076b076f076d0770076e0771077207730775077407760779077a07770778077b077f077c077d077e0781078007830782078707850784078607890788078d078a078b078e078c078f079107900793079207960794079707980795079c079f0799079d079a079e079b07a107a007a307a507a207a607a807a407a707af07aa07b207ab07a907b007ac07ae07ad07b107b307b507b907b707b807bd07b407b607bb07bc07ba07c007be07bf07c207c107c507c307c407c607c707ca07cb07c907cc07cd07ce07d207d107cf07c807d007d507d307d607d407ffff");
  std::string expected_target("007fffff00000000000000000000000000000000000000000000000000000000");

  byte *b2 = new byte[ShaiHive::SHAI_DATA_SIZE];
  hexstrToBytes(expected_input, b2);
  if (ShaiHive::hash(workCtx, b2))
  {
    printf("solved!\n");
  }
  delete[] b2;

  // std::string HASH = hexStr(ctx.sha, 32);

  std::cout << "expected hash: " << expected_hash << std::endl
            << std::flush;
  std::cout << "  actual hash: " << byteArrayToHexString(workCtx.sha, 32) << std::endl
            << std::flush;
  // std::cout << "  actual hash: " << workCtx.sha << std::endl << std::flush;

  std::string actual_path = convertPathToHexString(workCtx.path);
  if (expected_path_hex.length() == actual_path.length())
  {
    printf("Same size\n");
  }
  else
  {
    printf("%ld != %ld\n", expected_path_hex.length(), actual_path.length());
  }
  if (expected_path_hex == convertPathToHexString(workCtx.path))
  {
    std::cout << "path matches!" << std::endl
              << std::flush;
  }

  // hash_int: 23267840630650703295082657877313971011051472250179126958776604532498132026 < target_int: 226156397331686527036022285428078483006287265162934804098993112661845082112
  if (meets_target(expected_hash, expected_target))
  {
    std::cout << "meets_target!" << std::endl
              << std::flush;
  }

  // exit(0);

  // byte powHash2[32];
  // byte devWork[MINIBLOCK_SIZE*SHAI_BATCH];
  // byte work[MINIBLOCK_SIZE*SHAI_BATCH];

  // workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
  // initWorker(*worker);
  // lookupGen(*worker, nullptr, nullptr);

  // std::cout << *worker << std::endl;

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }

  srand(time(NULL));

  while (true)
  {
    try
    {
      boost::json::value myJob;
      boost::json::value myJobDev;
      boost::json::value *mineJob;
      {
        std::scoped_lock<boost::mutex> lockGuard(mutex);
        myJob = job;
        myJobDev = devJob;
        localJobCounter = jobCounter;
      }

      /*
      {
      "type":"job",
      "job_id":"816665",
      "data":"00000020e050b9a5c96a89b8b983a8b2cfcb098fe384bb02f1827f85f46b1eac060000003b140bce031610f6cb79b9288afb29777ccc56ef2e9ee6e0fa31d1ce4457df979f9b27676d60091d",
      "target":"007fffff00000000000000000000000000000000000000000000000000000000"
      }
      */

      // std::string job_id = std::string(myJob.at("job_id").as_string());
      // std::string target = std::string(myJob.at("target").as_string());
      // std::string data = std::string(myJob.at("data").as_string());

      // std::string nonceStr = std::string(myJob.at("nonce").as_string());

      byte *b2 = new byte[ShaiHive::SHAI_DATA_SIZE];
      hexstrToBytes(std::string(myJob.at("data").as_string()), b2);
      // std::cout << "Usr Job: " << std::string(myJob.at("job_id").as_string()) << " " << std::string(myJob.at("target").as_string()) << " " << std::string(myJob.at("data").as_string()) << std::endl
      //           << std::flush;
      memcpy(work, b2, ShaiHive::SHAI_DATA_SIZE);
      delete[] b2;

      // Num tgt = Num(target.c_str(), 16);
      // std::cout << "New Job: " << job_id << " " << target << " " << data << std::endl << std::flush;
      // std::cout << localJobCounter << " == " << jobCounter << std::endl << std::flush;

      if (devConnected)
      {
        byte *b2d = new byte[ShaiHive::SHAI_DATA_SIZE];
        hexstrToBytes(std::string(myJobDev.at("data").as_string()), b2d);
        // std::cout << "Dev Job: " << std::string(myJobDev.at("job_id").as_string()) << " " << std::string(myJobDev.at("target").as_string()) << " " << std::string(myJobDev.at("data").as_string()) << std::endl
        //           << std::flush;
        memcpy(devWork, b2d, ShaiHive::SHAI_DATA_SIZE);
        delete[] b2d;
      }

      /*
      if ((work[0] & 0xf) != 1)
      { // check  version
       //  mutex.lock();
        std::cerr << "Unknown version, please check for updates: "
                  << "version" << (work[0] & 0x1f) << std::endl;
       //  mutex.unlock();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        continue;
      }
      */

      double which;
      bool devMine = false;
      bool submit = false;
      int64_t DIFF;
      // difficulty = cmpDiff[0];
      //  DIFF = 5000;

      std::string hex;
      int32_t nonce = 0;
      // std::cout << localJobCounter << " == " << jobCounter << std::endl
      //           << std::flush;

      std::string target = std::string(myJob.at("target").as_string());
      std::string target_dev;
      if (devConnected)
        target_dev = std::string(myJobDev.at("target").as_string());

      uint8_t target_bytes[32];
      uint8_t target_bytes_dev[32];
      hexstrToBytes(target, target_bytes);
      hexstrToBytes(target_dev, target_bytes_dev);

      uint8_t * TT;

      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = (double)(rand() % 10000);
        devMine = (devConnected && which < devFee * 100.0);
        mineJob = devMine ? &myJobDev : &myJob;

        std::string job_id = std::string(mineJob->at("job_id").as_string());
        std::string data = std::string(mineJob->at("data").as_string());


        // printf("%s\n", target.c_str());


        byte *WORK = devMine ? &devWork[0] : &work[0];
        /*
        printf("before: 0x");
        for(int x = 0; x < ShaiHive::SHAI_DATA_SIZE; x++) {
          printf("%02x", WORK[x]);
        }
        printf("\n");
        */

        uint64_t *nonce = devMine ? &nonce0_dev : &nonce0;
        (*nonce)++;

        // uint64_t N = (uint64_t)nonce;
        // uint32_t trueN = getLeastSignificant32Bits(*nonce); //(N << 8) | tid;
        uint64_t N = (uint64_t)*nonce;
        uint32_t trueN = (N << 18) | ((tid & 511) << 10) | (rand() & 1023);
        memcpy(&WORK[76], &trueN, sizeof(trueN));
        /*
        printf("after : 0x");
        for(int x = 0; x < ShaiHive::SHAI_DATA_SIZE; x++) {
          printf("%02x", WORK[x]);
        }
        printf("\n");
        */
        // devMine = false;
        // DIFF = devMine ? difficultyDev : difficulty;

        // cmpDiff = Num(target.c_str(), 16);
        // std::cout << "target is: " << cmpDiff << std::endl;

        // printf("Difficulty: %" PRIx64 "\n", DIFF);

        // cmpDiff = ConvertDifficultyToBig(DIFF, DERO_HASH);

        // std::stringstream nonceAsHex;
        // nonceAsHex << std::hex << trueN;

        // std::string fullWork = data + nonceAsHex.str();
        // byte *workBytes = new byte[ShaiHive::SHAI_DATA_SIZE];
        // hexstrToBytes(fullWork, workBytes);

        // TODO FIXME DIRKER: whatever needs to happen!
        // do calculation with workBytes
        // currHash should contain the hash
        // pathHex should containt the path
        // byte *currHash = &powHash[0];
        // std::string pathHex = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

        if (ShaiHive::hash(workCtx, WORK))
        {
          counter.fetch_add(1);
          // std::string asdf(trueN);
          // std::hex(asdf);
          // printf("Not solved!\n");
          //}
          submit = devMine ? !submittingDev : !submitting;

          // std::reverse(workCtx.sha, workCtx.sha+32);
          // std::reverse(SHA, SHA+32);
          // std::string HASH = hexStr(SHA, 32);
          //  for (int i = 0; i < SHAI_BATCH; i++) {
          // byte *currHash = &workCtx.sha[0];
          // std::reverse(workCtx.sha, workCtx.sha + 32);

          // if (meets_target(hexStr(workCtx.sha, 32), target))
          TT = devMine ? target_bytes_dev : target_bytes;
          if (submit && ShaiHive::checkNonce((uint32_t*)workCtx.sha, (uint32_t*)TT))
          {
            uint32_t sN = __builtin_bswap32(trueN);
            // std::cout << "CheckHash was true!" << std::endl
            //           << std::flush;
            // std::cout << "nonce: " << uint32ToHex(sN) << std::endl
            //           << std::flush;

            // if (localJobCounter != jobCounter)
            // {
            //   // printf("submit %s || localJobCounter != jobCounter %s\n", (submit ? "true" : "false"), (localJobCounter != jobCounter ? "true" : "false"));
            //   // fflush(stdout);
            //   break;
            // }

            // printf("work: %s, hash: %s\n", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str(), hexStr(powHash, 32).c_str());
            // boost::lock_guard<boost::mutex> lock(mutex);
            /*
            let submit_msg = SubmitMessage {
              r#type: String::from("submit"),
              miner_id: miner_id.to_string(),
              nonce: nonce,
              job_id: job.job_id.clone(),
              path: path_hex,
              };
            */
           
            std::string pathHex = convertPathToHexString(workCtx.path);
            if (devMine)
            {
              submittingDev = true;
              setcolor(CYAN);
              std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
              setcolor(BRIGHT_WHITE);
              devShare = {
                  {"type", "submit"},
                  {"miner_id", walletDev.c_str()}, // devSelection[SHAI_COIN]},
                  {"nonce", uint32ToHex(sN).c_str()},
                  {"job_id", job_id},
                  {"path", pathHex.c_str()}};
              data_ready = true;
            }
            else
            {

              submitting = true;
              setcolor(BRIGHT_YELLOW);
              std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
              setcolor(BRIGHT_WHITE);
              share = {
                  {"type", "submit"},
                  {"miner_id", wallet.c_str()},
                  {"nonce", uint32ToHex(sN).c_str()},
                  {"job_id", job_id},
                  {"path", pathHex.c_str()}};
              data_ready = true;
            }
            cv.notify_all();

            //} else {
            //  printf("Check hash failed for nonce %s\n", nonceAsHex.str().c_str());
            //  fflush(stdout);
          }
        } // ShaiHive::hash
        if (!isConnected)
          break;
      }
      if (!isConnected)
        break;
    }
    catch (std::exception &e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl
                << std::flush;
      setcolor(BRIGHT_WHITE);

      localJobCounter = -1;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}
