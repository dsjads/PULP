

package net.sf.zipme;
class Deflater {

   public int getTotalIn(){
    return (int)engine.getTotalIn();
  }

   public long getBytesRead(){
    return engine.getTotalIn();
  }

   public long getBytesWritten(){
    return totalOut;
  }

   void flush(){
    state|=IS_FLUSHING;
  }

   public void finish(){
    state|=IS_FLUSHING | IS_FINISHING;
  }

   public boolean finished(){
    return state == FINISHED_STATE && pending.isFlushed();
  }

   public boolean needsInput(){
    return engine.needsInput();
  }

   public void setInput(  byte[] input){
    setInput(input,0,input.length);
  }

   public void setInput(  byte[] input,  int off,  int len){
    if ((state & IS_FINISHING) != 0)     throw new IllegalStateException("finish()/end() already called");
    engine.setInput(input,off,len);
  }

   public void setLevel(  int lvl){
    if (lvl == DEFAULT_COMPRESSION)     lvl=6;
 else     if (lvl < NO_COMPRESSION || lvl > BEST_COMPRESSION)     throw new IllegalArgumentException();
    if (level != lvl) {
      level=lvl;
      engine.setLevel(lvl);
    }
  }

   public void setStrategy(  int stgy){
    if (stgy != DEFAULT_STRATEGY && stgy != FILTERED && stgy != HUFFMAN_ONLY)     throw new IllegalArgumentException();
    engine.setStrategy(stgy);
  }

   public int deflate(  byte[] output){
    return deflate(output,0,output.length);
  }

   public int deflate(  byte[] output,  int offset,  int length){
    return new Deflater_deflate2(this,output,offset,length).execute();
  }

   public void setDictionary(  byte[] dict){
    setDictionary(dict,0,dict.length);
  }

   public void setDictionary(  byte[] dict,  int offset,  int length){
    if (state != INIT_STATE)     throw new IllegalStateException();
    state=SETDICT_STATE;
    engine.setDictionary(dict,offset,length);
  }
   protected void hook24(  int lvl){
    setStrategy(DEFAULT_STRATEGY);
    setLevel(lvl);
    original(lvl);
  }
   protected void hook25(){
    engine=new DeflaterEngine(pending);
    original();
  }

   public void reset(){
    original();
    engine.reset();
  }

   public void end(){
    engine=null;
    original();
  }
}
