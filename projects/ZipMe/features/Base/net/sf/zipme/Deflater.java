

package net.sf.zipme;


public class Deflater {

  public static final int BEST_COMPRESSION=9;

  public static final int BEST_SPEED=1;

  public static final int DEFAULT_COMPRESSION=-1;

  public static final int NO_COMPRESSION=0;

  public static final int DEFAULT_STRATEGY=0;

  public static final int FILTERED=1;

  public static final int HUFFMAN_ONLY=2;

  public static final int DEFLATED=8;
  public static final int IS_SETDICT=0x01;
  public static final int IS_FLUSHING=0x04;
  public static final int IS_FINISHING=0x08;
  private static final int INIT_STATE=0x00;
  private static final int SETDICT_STATE=0x01;
  private static final int INIT_FINISHING_STATE=0x08;
  private static final int SETDICT_FINISHING_STATE=0x09;
  public static final int BUSY_STATE=0x10;
  public static final int FLUSHING_STATE=0x14;
  public static final int FINISHING_STATE=0x1c;
  public static final int FINISHED_STATE=0x1e;
  public static final int CLOSED_STATE=0x7f;

  public int level;

  public boolean noHeader;

  public int state;

  public long totalOut;

  public DeflaterPending pending;

  public DeflaterEngine engine;

  public Deflater(){
    this(DEFAULT_COMPRESSION,false);
  }

  public Deflater(  int lvl){
    this(lvl,false);
  }

  public Deflater(  int lvl,  boolean nowrap){
    if (lvl == DEFAULT_COMPRESSION)     lvl=6;
 else     if (lvl < NO_COMPRESSION || lvl > BEST_COMPRESSION)     throw new IllegalArgumentException();
    pending=new DeflaterPending();
    this.hook25();
    this.noHeader=nowrap;
    this.hook24(lvl);
    reset();
  }

  public void reset(){
    state=(noHeader ? BUSY_STATE : INIT_STATE);
    totalOut=0;
    pending.reset();
  }

  public void end(){
    pending=null;
    state=CLOSED_STATE;
  }

  public int getTotalOut(){
    return (int)totalOut;
  }
  protected void hook24(  int lvl){
  }
  protected void hook25(){
  }
}
