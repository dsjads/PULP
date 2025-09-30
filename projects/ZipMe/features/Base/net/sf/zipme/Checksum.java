

package net.sf.zipme;


public interface Checksum {

  long getValue();

  void reset();

  void update(  int bval);

  void update(  byte[] buf,  int off,  int len);
}
